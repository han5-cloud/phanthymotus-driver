"""
unitree/r1/speaker_raise.py — 「先说话、再举手」组合卡片（actuator）。

为什么是驱动里的一张新卡片，而不是画布上 tts + arm 两张卡片连着用：
  - tts.speak 调的是 AudioClient.TtsMaker，RPC 返回时语音还在播，且 tts 没有声明
    x-completion。agent-core 的 ACP barrier 因此不会等它，LLM 紧接着调 arm 时，
    举手会和播报同时发生，做不到"先说完再举手"。
  - R1 的 audio RPC（1001-1010）里没有"播放结束"查询，所以本卡片只能按字数估算
    等待时间（speak_wait_s 可手动覆盖）。这是估算，不是播放结束检测。

为什么必须放进 r1-device-bundle，而不是单独起一个 bundle：
  - ArmClient 以 lease=True 构造（unitree_sdk2py/r1/arm/r1_arm_client.py），
    arm 服务的租约同一时刻只有一个持有者。另起一个进程再建 ArmClient，会和
    r1-device-bundle 的 RpcProxy 抢租约。这里复用同一个 RpcProxy。

参数名用 `speech` 而不是 `text`：agent-core 在参数里有 `text` 时会把 ACP 超时
改成 len(text)/3+10 秒（mcp_client.py 的 dynamic_timeout），覆盖 schema 里的
timeout。本卡片的总时长 = 播报 + 手势保持 + 放下，远长于这个公式，用 `text`
会让 barrier 在动作没做完时就超时放行。
"""

import json
import os
import threading
import time
from uuid import uuid4

ARM_RELEASE_ID = 99                 # release_arm：把手放下（device.py 同样用 99）
GROUND_OR_TRANSIT = {0, 1, 701, 702}  # 躺着/阻尼/起身中/躺下中：不做手势
DEFAULT_SPEECH = "欢迎来到范式智能"
RAISE_GESTURES = ("right_hand_up", "both_hands_up", "wave_above_head")


def _post_acp(action_id: str, status: str, result: dict, tool: str) -> None:
    """POST ACP 完成回执。失败只打日志——barrier 会在 x-completion.timeout 后放行。"""
    import ssl
    import urllib.request

    url = os.environ.get("AGENT_CORE_URL", "https://localhost:15678")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    body = json.dumps({"action_id": action_id, "status": status, "result": result,
                       "tool": tool, "ts": time.time()}).encode()
    try:
        req = urllib.request.Request(f"{url}/api/acp/complete", data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        urllib.request.urlopen(req, timeout=5, context=ctx)
        print(f"[speaker_raise] ACP {action_id} -> {status}", flush=True)
    except Exception as e:
        print(f"[speaker_raise] ACP notify FAILED {action_id}: {e}", flush=True)


class SpeakerRaisePlugin:
    PREFIX = "speaker_raise"

    def __init__(self, plugin_config: dict, rpc, arm_actions: dict,
                 loco_plugin=None, notifier=_post_acp, sleep=time.sleep):
        cfg = plugin_config or {}
        # 考题要求卡片名是 "speaker raise"（带空格）。但 agent-core 直接把它拼成
        # function 名 mcp__<id>__speaker raise 发给模型，而 OpenAI/Anthropic 的
        # function 名只允许 [a-zA-Z0-9_-]。只在控制页手动点"执行"时空格没问题。
        self._name = cfg.get("tool_name", "speaker_raise")
        self._rpc = rpc
        self._arm = {k: v for k, v in arm_actions.items() if k in RAISE_GESTURES}
        self._loco = loco_plugin
        self._notify = notifier
        self._sleep = sleep
        self._chars_per_s = float(cfg.get("chars_per_s", 4.0))   # 需在真机上标定
        self._speak_margin = float(cfg.get("speak_margin_s", 0.8))
        self._require_upright = bool(cfg.get("require_upright", True))
        self._busy = threading.Lock()
        self._cancel = threading.Event()

    # ── schema ────────────────────────────────────────────────────────────
    def get_tool(self) -> dict:
        return {
            "name": self._name,
            "type": "actuator",
            "multiInstance": False,
            "description": "先用机载 TTS 播报一句话，等播报结束（按字数估算）后举手，"
                           "保持 hold_s 秒后放下。机器人躺下或正在起身/躺下时拒绝执行。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["run", "stop"]},
                    "speech": {"type": "string",
                               "description": f"播报内容，留空为「{DEFAULT_SPEECH}」"},
                    "gesture": {"type": "string", "enum": sorted(self._arm),
                                "description": "举手动作，默认 right_hand_up"},
                    "voice": {"type": "integer", "description": "0=中文 1=英文，默认 0"},
                    "speak_wait_s": {"type": "number",
                                     "description": "播报后等待秒数；留空按字数估算"},
                    "hold_s": {"type": "number", "description": "举手保持秒数，默认 4"},
                },
                "required": ["action"],
                "x-resource": ["mouth", "arm"],
                "x-action-params": {
                    "run": {"params": ["speech", "gesture", "voice", "speak_wait_s", "hold_s"],
                            "description": "先说话再举手"},
                    "stop": {"params": [], "description": "中断并放下手"},
                },
                "x-completion": {"actions": ["run"], "timeout": 60},
            },
        }

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._cancel.set()

    # ── dispatch ──────────────────────────────────────────────────────────
    def dispatch(self, action: str, args: dict) -> dict | None:
        if action == "start":
            return {"state": "ready"}
        if action == "info":
            return {"state": "busy" if self._busy.locked() else "ready"}
        if action == "stop":
            if not self._busy.locked():
                return {"state": "idle"}
            self._cancel.set()
            code, _ = self._rpc.ArmStop()
            return {"state": "stopping", "arm_stop_ret": code}
        if action != "run":
            return {"error": f"unknown action: {action}"}

        speech = (args.get("speech") or "").strip() or DEFAULT_SPEECH
        gesture = args.get("gesture") or "right_hand_up"
        if gesture not in self._arm:
            return {"error": f"gesture must be one of {sorted(self._arm)}"}
        try:
            voice = int(args.get("voice", 0) or 0)
            hold_s = max(1.0, min(10.0, float(args.get("hold_s") or 4.0)))
            wait = args.get("speak_wait_s")
            speak_wait = (float(wait) if wait not in (None, "")
                          else len(speech) / self._chars_per_s + self._speak_margin)
        except (TypeError, ValueError) as e:
            return {"error": f"bad argument: {e}"}

        # 前置检查放在说话之前：先说了再拒绝举手，用户听到的是半个动作。
        refusal = self._preflight()
        if refusal:
            return {"error": refusal}
        if not self._busy.acquire(blocking=False):
            return {"error": "speaker_raise is already running"}

        self._cancel.clear()
        action_id = f"r1_spkraise_{uuid4().hex[:8]}"
        threading.Thread(target=self._worker, daemon=True,
                         args=(action_id, speech, voice, gesture, speak_wait, hold_s)).start()
        return {"status": "executing", "action_id": action_id, "speech": speech,
                "gesture": gesture, "speak_wait_s": round(speak_wait, 2), "hold_s": hold_s}

    def _preflight(self) -> str | None:
        if self._loco is not None and getattr(self._loco, "_fsm_active", None):
            return f"posture change {self._loco._fsm_active} in progress"
        if not self._require_upright:
            return None
        code, fsm = self._rpc.GetFsmId()
        if code != 0:
            return f"cannot read FSM (code={code}); refusing to move arms"
        if fsm in GROUND_OR_TRANSIT:
            return f"robot not upright (fsm={fsm}); stand it up first"
        return None

    # ── worker ────────────────────────────────────────────────────────────
    def _worker(self, action_id, speech, voice, gesture, speak_wait, hold_s):
        stages, status, raised = [], "completed", False
        t0 = time.monotonic()
        try:
            code = self._rpc.TtsMaker(speech, voice)
            stages.append({"stage": "tts", "code": code})
            if code != 0:
                status = "error"
                return
            self._sleep(speak_wait)
            if self._cancel.is_set():
                status = "cancelled"
                return

            en_code, _ = self._rpc.ArmEnable()
            # 上游 arm 工具忽略 Enable 的返回码；这里记录下来，以 Execute 的码为准。
            stages.append({"stage": "enable", "code": en_code})
            ex_code, _ = self._rpc.ArmExecuteById(self._arm[gesture])
            stages.append({"stage": "gesture", "name": gesture, "code": ex_code})
            if ex_code != 0:
                status = "error"
                return
            raised = True
            self._sleep(hold_s)
            if self._cancel.is_set():
                status = "cancelled"
        except Exception as e:
            stages.append({"stage": "exception", "error": f"{type(e).__name__}: {e}"})
            status = "error"
        finally:
            if raised or self._cancel.is_set():
                rel_code, _ = self._rpc.ArmExecuteById(ARM_RELEASE_ID)
                stages.append({"stage": "release", "code": rel_code})
            self._busy.release()
            result = {"stages": stages, "elapsed_s": round(time.monotonic() - t0, 2)}
            print(f"[speaker_raise] {action_id} {status} {result}", flush=True)
            self._notify(action_id, status, result, self._name)
