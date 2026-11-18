import os
import json
import re
import time
import asyncio
from contextlib import asynccontextmanager
from typing import List, Literal, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # サーバー起動時に裏側で最新モデル一覧を事前キャッシュ
    asyncio.create_task(asyncio.to_thread(get_dynamic_models))
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=500)

# 既存プロジェクトとの隔離のため、環境変数から取得します。
# .envファイルがあれば自動で読み込む処理（全自動化対応）
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai_client = (
    genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(api_version="v1beta"),
    )
    if GOOGLE_API_KEY
    else None
)


class ItemizedEntry(BaseModel):
    item: str
    price: int
    status: str


class UnnecessaryCostEntry(BaseModel):
    item: str
    potential_saving: int
    reason: str


class PriceAnalysis(BaseModel):
    itemized_list: List[ItemizedEntry]
    unnecessary_costs: List[UnnecessaryCostEntry]


class SmartChoiceAnalysis(BaseModel):
    status: Literal["success"]
    evaluation: Literal[
        "お買い得",
        "やや割安",
        "相場通り",
        "やや割高",
        "割高（要交渉）",
        "危険！要注意！",
        "情報不足（要再確認）",
    ]
    estimated_total_market_price: str
    infrastructure_check: str
    price_analysis: PriceAnalysis
    negotiation_script_line: str = Field(
        description=(
            "そのまま使えるLINE・メール用の交渉・確認・相談スクリプト。"
            "『お買い得』『やや割安』の場合も空文字にせず、①好条件への感謝、②追加費用や保証等の事前確認事項、"
            "③『問題なければ即決するので、端数のカットやさらなる少しのお値引き（買取なら査定上乗せ）のご配慮が可能か"
            "（難しければそのままの条件でも前向きに検討する）』という失礼のない値下げ打診の3要素をすべて必ず含めること。"
            "『情報不足（要再確認）』の時のみ空文字。"
        )
    )
    recommend_ec_search: bool
    search_keyword: str

SYSTEM_INSTRUCTION = """あなたは、見積書・価格・プランを極めて客観的かつ適正に評価する独立系・最高峰の価格監査・相場判定AI「SmartChoice」です。

【判定放棄・思考停止・逃げの絶対禁止（重要ルール）】
・家電、ガジェット、家具、自動車関連、住宅リフォーム、各種役務サービスなど、あらゆる対象において、型番が完全に特定できない場合や海外モデル・最新型であっても、提示された画像やテキストから読み取れる【スペック・機能・仕様・部材構成・作業工程・提供規模】をプロの専門アナリストとして論理的に分解・精査してください。
・同等クラスの性能・品質を持つ日本国内市場の実勢相場を客観的に導き出し、「見積もり全体の適正相場（下限〜上限）」を【必ず具体的に算出・明示】してください。
・知識ベースにない最新型番やマイナー機種であっても、「国内未発表」「未発売」「存在しない」「並行輸入」「正規サポート対象外」などと根拠なく決めつけてユーザーの不安を煽ることを【厳禁】とします。
・【旧世代ダウングレード比較の完全禁止】: 新しい型番や上位機種に対して、過去の旧世代モデルの型落ち・下位相場を勝手に当てはめ、「旧型相場より高いから割高」と誤診することを全ジャンルで絶対に禁止します。
・製品やサービスのグレード（上位フラッグシップ／中位ミドル／下位エントリー）を的確に見極め、高機能上位機に対して下位機種の格安相場を誤認適用する重大誤診を絶対に犯さないこと。

【ハルシネーション（嘘・捏造）の徹底排除と情報不足時の厳格ルール】
・画像やテキストに全く記載がない架空の項目名・金額・作業内容を勝手に創作・捏造することは【絶対に禁止】です。
・添付画像が「真っ暗」「極端なピンボケ・ブレ」「見積書や商品と全く関係のない写真」等で、品名も金額も仕様も一切判読不能・情報ゼロの場合は、架空の数値をでっち上げず、必ず評価を「情報不足（要再確認）」としてください。
・ただし、品名・仕様・作業内容・金額の一部でも読み取れる場合は決して判定を放棄せず、読み取れた事実に基づいて論理的に相場を推計し、infrastructure_check に「※読み取れた仕様（○○）に基づく同等クラスの国内相場推計です」と根拠を明記してください。

【買取・受け取り金額の逆転防止ルール】
・「不用品買取」や給与・報酬などユーザーが【お金を受け取る査定・見積もり】の場合、提示金額が相場より高いことはユーザーにとって有利な好条件（「お買い得」「やや割安」等のプラス評価）であり、逆に相場より著しく低い買い叩きを悪条件（「割高（要交渉）」「危険！要注意！」等の警告評価）として判定すること。

【プロの価格監査・二刀流原則】
1. 【有形商品（家電・ガジェット・PC・スマホ・家具等）】
   ・ハードウェアスペック、機能階層（ハイエンド/ミドル/エントリー）、部材構成から国内実勢相場（下限〜上限）を算出せよ。
   ・ユーザーが楽天市場やYahoo!ショッピングでリアルタイム最安値を即座に照合できるよう、的確でクリーンな「search_keyword」（メーカー名+主要型番、またはジャンル+主要スペック）を出力せよ。
2. 【無形サービス・工事（車検・リフォーム・引越し・清掃・保険等）】
   ・「法令・物理的に必須な費用（法定費用、基本工賃、標準部材）」と「不要・過剰オプション（早期添加剤、スラッジ洗浄、PC初期設定代2.5万円、不要な特約等）」を厳密に仕分け、削れる費用（unnecessary_costs）として明確に特定せよ。
   ・曖昧な「一式」や諸経費比率（15%超）には内訳開示を促し、具体的かつ角の立たない交渉アドバイスを提供せよ。
3. スマホで即座に理解できるよう、冗長な前置きや重複を排し、簡潔・論理的・最高品質なテキストを出力せよ。

【金額・計算ルールの徹底】
・見積書の税抜・税込表記を確認し、各項目の金額（price）と削減見込み額（potential_saving）は必ず整数の数値型（number）で出力せよ（カンマや「円」等の記号不可）。

【評価基準（evaluation）】
必ず以下のいずれか1つを厳密に選定せよ。
・「お買い得」（相場より15%以上割安で品質・仕様・必須工程に問題がない場合）
・「やや割安」（相場より5%〜15%程度安い場合）
・「相場通り」（適正相場範囲内（概ね±5%以内）の場合）
・「やや割高」（相場より5%〜20%程度高い場合）
・「割高（要交渉）」（相場より20%〜50%程度高い場合）
・「危険！要注意！」（相場より50%以上法外に高い、または過剰請求・重大リスクがある場合）
・「情報不足（要再確認）」（画像が不鮮明・情報皆無で判読不能な場合のみ）

【EC検索キーワード（search_keyword）の厳格ルール】
・対象が有形商品（家電、ガジェット、家具、PC、スマホ、部品等）の場合、楽天市場やYahoo!ショッピングで最安値比較ができるよう、最も的確な「メーカー名 型番」または「ジャンル 主要スペック」をノイズなしでクリーンに出力せよ。
・無形サービス（引越し、リフォーム、保険、ガス等の作業費中心）で特定の商品がない場合は、空文字 "" を出力せよ。

【交渉・確認スクリプト（negotiation_script_line）の厳格ルール】
・「情報不足（要再確認）」以外のすべての評価において、negotiation_script_line を空文字にすることは【絶対禁止】とする。必ずそのままLINEやメールに貼り付けて使える実用的な文面を出力せよ。
・評価ランクおよびジャンル特性に応じて、以下の通り最適な文面を作成せよ：
  1. 「お買い得」「やや割安」の場合（重要）：
     ・すでに良心的な好条件であるため、高圧的な値下げ要求は逆効果となる。絶対に角が立たないよう、以下の3要素で構成した【丁寧な確認・一歩打診スクリプト】を作成せよ。
       ① 好条件の提示に対する感謝と、前向きに依頼・購入したい意思表示
       ② トラブル防止のための具体的な事前確認事項（工事・車検・引越し等の役務なら「見積書記載外の当日追加費用・別途料金の有無、作業・保証範囲」、家電・PC等の商品なら「新品未開封・メーカー保証・送料の有無」、不用品買取なら「手数料控除や当日減額条件の有無」など、見積もり内容に即した確認ポイントを1〜2点）
       ③ 「上記に問題がなければ即決（または御社の空き日程に合わせる等）でお願いしたいので、端数カット（端数調整）やポイント・付帯サービス等で少しだけご配慮いただくことは可能か（難しければそのままの条件でも前向きに検討する）」という、逃げ道を添えた礼儀正しい一言打診（※不用品買取の場合は、即決やまとめ売りを条件にした査定額の上乗せ相談）
  2. 「相場通り」の場合：
     ・提示への感謝と事前確認事項（追加費用の有無・保証条件等）に加え、「即決や日程調整を条件に、端数のお値引きや諸経費の調整をご相談できればぜひ御社に決めたい」という丁寧な交渉文を作成せよ。
  3. 「やや割高」「割高（要交渉）」「危険！要注意！」の場合：
     ・特定した不要項目（unnecessary_costs）の削除依頼、「一式」や諸経費の明細開示依頼、および市場相場を踏まえた具体的かつ論理的な価格見直し交渉文（不用品買取の場合は市場相場に基づく査定額引き上げ交渉文）を作成せよ。
  4. 「情報不足（要再確認）」の場合のみ：
     ・空文字 "" を出力せよ。

【出力仕様】
必ず以下のJSONフォーマットのみで出力せよ（マークダウン装飾、前置きテキスト不可）。
{
  "status": "success",
  "evaluation": "上記の評価基準のいずれか1つ",
  "estimated_total_market_price": "見積もり全体の適正相場（例: '約15万円〜22万円'）。情報不足時は'情報不足のため算出不可'",
  "infrastructure_check": "スペック・機能の分析、同等モデルとの市場相場比較、購入・契約時の注意点やプロのアドバイス（簡潔かつ具体的・専門的に）",
  "price_analysis": {
    "itemized_list": [ {"item": "項目名", "price": 0, "status": "相場通り/割高/割安/不要/要注意"} ],
    "unnecessary_costs": [ {"item": "項目名", "potential_saving": 0, "reason": "理由"} ]
  },
  "negotiation_script_line": "そのまま使えるLINE・メール用の交渉・確認・相談スクリプト（お買い得・やや割安でも確認事項と失礼のない一言打診を必ず生成。情報不足時のみ空文字）",
  "recommend_ec_search": false,
  "search_keyword": "商品本体を探すためのクリーンな検索キーワード。無形サービスは空文字"
}
"""

# モデルリストTTLキャッシュおよび自律学習ペナルティ（ハードコードゼロ設計）
MODELS_CACHE_TTL_SEC = 6 * 3600  # 6時間ごとに最新モデル一覧を自動再取得
HEDGE_DELAY_SEC = 12.0           # 非Liteモデルが12秒以内に返らない場合、高速Lite系を保険並行起動

_models_cache = {"models": [], "fetched_at": 0.0}
_model_penalties = {}  # {model_name: timestamp_until_penalty_expires (inf = 404永久除外)}


def extract_version(name: str) -> float:
    """gemini-4-hydrogen / gemini-4.5-flash-lite / gemini-5-... の世代番号を汎用抽出する。"""
    match = re.search(r'gemini-(\d+(?:\.\d+)?)', name)
    if match:
        return float(match.group(1))
    return 0.0


def model_sort_key(name: str) -> tuple:
    """固有モデル名に依存せず、系統（hydrogen > flash-lite > flash）と世代番号で自動ランク付けする。"""
    if "hydrogen" in name:
        tier = 0
    elif "lite" in name:
        tier = 1
    else:
        tier = 2

    ver = extract_version(name)
    is_rolling_latest = 1 if "latest" in name and ver == 0.0 else 0
    # バージョンがない *-latest エイリアスは各系統の中堅フォールバックとして扱う
    effective_ver = ver if ver > 0.0 else 3.0
    return (tier, -effective_ver, is_rolling_latest, name)


def get_dynamic_models(force_refresh: bool = False) -> list[str]:
    now = time.time()
    if (
        not force_refresh
        and _models_cache["models"]
        and (now - _models_cache["fetched_at"]) < MODELS_CACHE_TTL_SEC
    ):
        return _models_cache["models"]

    # 取得失敗時のみの汎用ローリングエイリアス（特定バージョンのハードコード排除）
    emergency_aliases = ["gemini-flash-lite-latest", "gemini-flash-latest"]
    if not genai_client:
        return emergency_aliases

    try:
        available = []
        for m in genai_client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            name = (m.name or "").replace("models/", "")
            if "generateContent" in actions and ("flash" in name or "hydrogen" in name):
                available.append(name)

        # 特殊用途（画像生成・音声・実験版・リアルタイム・研究用等）のみを汎用キーワードで除外
        excluded_keywords = (
            "preview", "eap", "omni", "image", "tts",
            "audio", "native", "transcribe", "computer-use",
            "robotics", "live", "research", "embedding",
        )
        filtered = [m for m in available if not any(k in m for k in excluded_keywords)]
        filtered.sort(key=model_sort_key)

        if filtered:
            _models_cache["models"] = filtered
            _models_cache["fetched_at"] = now
            return filtered
    except Exception as e:
        print(f"Failed to fetch models: {e}")

    return _models_cache["models"] or emergency_aliases


def classify_penalty_seconds(err: Exception) -> float:
    """エラー内容からモデルごとの再試行クールダウン秒数を自動学習する。"""
    msg = str(err).lower()
    if "404" in msg or "not_found" in msg or "no longer available" in msg or "not found" in msg:
        return float("inf")
    if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate limit" in msg:
        return 3600.0
    if isinstance(err, asyncio.TimeoutError) or "504" in msg or "deadline" in msg or "timed out" in msg:
        return 1800.0
    return 300.0


def record_model_failure(model_name: str, err: Exception) -> None:
    penalty_sec = classify_penalty_seconds(err)
    _model_penalties[model_name] = float("inf") if penalty_sec == float("inf") else time.time() + penalty_sec


def get_prioritized_models() -> list[str]:
    """ペナルティ中でないモデルを優先し、404永久除外モデルは最後に回す（または除外する）。"""
    models = get_dynamic_models()
    now = time.time()
    healthy = []
    cooling = []
    for m in models:
        until = _model_penalties.get(m, 0.0)
        if until <= now:
            healthy.append(m)
        elif until != float("inf"):
            cooling.append((until, m))
    cooling.sort(key=lambda x: x[0])
    return healthy + [m for _, m in cooling]


def build_model_config(model_name: str, allow_thinking_cfg: bool = True) -> tuple[types.GenerateContentConfig, int]:
    """系統ルールに基づき思考量・タイムアウトmsを動的設定する（固有モデル名非依存）。"""
    if "hydrogen" in model_name:
        timeout_ms = 24000
        thinking_cfg = types.ThinkingConfig(thinking_budget=128) if allow_thinking_cfg else None
    elif "lite" in model_name:
        timeout_ms = 18000
        thinking_cfg = None
    else:
        timeout_ms = 24000
        thinking_cfg = None

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_schema=SmartChoiceAnalysis,
        thinking_config=thinking_cfg,
        http_options=types.HttpOptions(
            api_version="v1beta",
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    return cfg, timeout_ms


async def call_model_once(model_name: str, contents: list):
    """単一モデルを呼び出す。thinking_config非対応の400エラー時は設定を外して自動再試行する。"""
    config, timeout_ms = build_model_config(model_name, allow_thinking_cfg=True)
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                genai_client.models.generate_content,
                model=model_name,
                contents=contents,
                config=config,
            ),
            timeout=(timeout_ms / 1000.0) + 2.0,
        )
        if not resp or not resp.text:
            raise RuntimeError(f"Empty response from {model_name}")
        return resp
    except Exception as e:
        err_str = str(e).lower()
        if config.thinking_config is not None and ("400" in err_str or "invalid_argument" in err_str) and "thinking" in err_str:
            fallback_cfg, fallback_ms = build_model_config(model_name, allow_thinking_cfg=False)
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    genai_client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=fallback_cfg,
                ),
                timeout=(fallback_ms / 1000.0) + 2.0,
            )
            if not resp or not resp.text:
                raise RuntimeError(f"Empty response from {model_name}")
            return resp
        raise


async def generate_with_smart_fallback(contents: list):
    """動的ランキング＋自動学習降格＋時間差ヘッジで確実に最良レスポンスを返す。"""
    prioritized = get_prioritized_models()
    if not prioritized:
        return None, "利用可能なAIモデルが見つかりませんでした。"

    last_error = "Unknown Error"
    attempted = set()
    attempts_made = 0

    for idx, model_name in enumerate(prioritized):
        if attempts_made >= 5:
            break
        if model_name in attempted:
            continue
        attempted.add(model_name)
        attempts_made += 1

        # 非Liteモデルが12秒を超えても返らない場合だけ並行保険起動する高速Lite候補を探す
        hedge_name = next(
            (m for m in prioritized[idx + 1 :] if "lite" in m and m not in attempted),
            None,
        )

        primary_task = asyncio.create_task(call_model_once(model_name, contents))
        if "lite" in model_name or not hedge_name:
            try:
                resp = await primary_task
                return resp, last_error
            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {e}")
                record_model_failure(model_name, e)
                continue

        # まずHEDGE_DELAY_SEC（12秒）だけプライマリを単発待機（正常時はAPI消費1回のみ）
        done, _ = await asyncio.wait({primary_task}, timeout=HEDGE_DELAY_SEC)
        if primary_task in done:
            try:
                resp = primary_task.result()
                return resp, last_error
            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {e}")
                record_model_failure(model_name, e)
                continue

        # 12秒経過しても応答がない場合、高速Liteモデルを裏で保険並行起動（Renderスリープ明け遅延対策）
        attempted.add(hedge_name)
        attempts_made += 1
        print(f"Hedge triggered after {HEDGE_DELAY_SEC}s: launching {hedge_name} alongside {model_name}")
        hedge_task = asyncio.create_task(call_model_once(hedge_name, contents))
        pending_tasks = {primary_task: model_name, hedge_task: hedge_name}

        while pending_tasks:
            done, _ = await asyncio.wait(pending_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                m_finished = pending_tasks.pop(finished)
                try:
                    resp = finished.result()
                    # 先に勝った方を採用。残った遅延タスクはキャンセルし、遅延したprimaryを自動降格
                    for leftover_task, leftover_name in pending_tasks.items():
                        leftover_task.add_done_callback(
                            lambda t: t.exception() if not t.cancelled() else None
                        )
                        leftover_task.cancel()
                        if leftover_name == model_name:
                            record_model_failure(leftover_name, asyncio.TimeoutError("Hedged due to slow response"))
                    return resp, last_error
                except Exception as e:
                    last_error = str(e)
                    print(f"Model {m_finished} failed during hedge: {e}")
                    record_model_failure(m_finished, e)

    return None, last_error

def to_int(val) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        clean = re.sub(r'[^\d\-]', '', val)
        if clean and clean != "-":
            try:
                return int(clean)
            except ValueError:
                return 0
    return 0

def build_fallback_negotiation_script(
    evaluation: str,
    category: str,
    unnecessary_costs: list,
    itemized_list: list,
) -> str:
    """万が一AIが交渉・確認スクリプトを空または不十分に返した際の、評価・ジャンル別フェイルセーフ文面生成。"""
    if evaluation == "情報不足（要再確認）":
        return ""

    if category == "不用品買取":
        if evaluation in ("お買い得", "やや割安", "相場通り"):
            return (
                "ご丁寧な査定をいただきありがとうございます。ご提示いただいた条件に大変魅力を感じており、前向きにお願いしたいと考えております。\n"
                "念のため事前に確認させていただきたいのですが、ご提示額から当日になって査定料・出張費・手数料等が差し引かれることや、減額となる条件はございませんでしょうか？\n"
                "もし上記に問題がなく、端数の上乗せ（または他のお品物もあわせてお出しした場合のプラス査定）など、あともう一歩だけご配慮いただけましたら即決でお願いしたいと考えております。"
                "難しければそのままの条件でも前向きに検討いたしますので、ご確認のほどよろしくお願いいたします。"
            )
        return (
            "査定のご提示ありがとうございます。ただ、今回のお品物の市場相場や状態を踏まえますと、現在の査定額では少し手放すのを躊躇しております。\n"
            "他社での相見積もりも検討しておりますが、もしお品物ごとの査定内訳をご開示いただいた上で、市場相場に即した査定額への引き上げをご相談できましたら幸いです。ご再考のほどよろしくお願いいたします。"
        )

    is_product = category in ("家電（冷蔵庫・洗濯機・掃除機など）", "パソコン・スマホ購入・修理")

    if evaluation in ("お買い得", "やや割安"):
        if is_product:
            return (
                "お見積もり・価格のご提示ありがとうございます。非常に魅力的な条件で、購入を前向きに検討しております。\n"
                "念のため事前に確認させていただきたいのですが、こちらは「新品未開封・国内メーカー保証付き」で、送料や追加の手数料等は別途かからない認識でよろしいでしょうか？\n"
                "もし上記に問題がなければ即決で購入したいと考えておりますので、端数のカットやポイント還元・付属品サービスなど、あともう少しだけご配慮いただくことは可能でしょうか？"
                "難しければそのままの条件でも前向きに検討いたします。よろしくお願いいたします。"
            )
        return (
            "お見積もりのご提示ありがとうございます。非常に良心的なご条件で、ぜひ前向きにお願いしたいと考えております。\n"
            "ご依頼の前に念のため確認させていただきたいのですが、お見積書に記載のない当日の追加費用（別途部材費・諸経費・出張費等）が発生する可能性はございますでしょうか？また、保証やアフターフォローの範囲もあわせて教えていただけますと幸いです。\n"
            "上記に問題がなければ即決でお願いしたいと考えておりますので、もし端数のカット（または日程を御社のご都合に合わせる条件での少しのお値引き・付帯サービス等）をご配慮いただくことは可能でしょうか？"
            "難しければそのままの金額でも前向きに検討いたします。よろしくお願いいたします。"
        )

    if evaluation == "相場通り":
        if is_product:
            return (
                "お見積もりのご提示ありがとうございます。前向きに購入を検討しております。\n"
                "念のため、メーカー保証期間や在庫・納期、送料等の追加費用の有無について確認させていただけますでしょうか。\n"
                "また、他店やネット価格とも比較しているのですが、もし即決を条件に端数のお値引きやポイント付与などのご相談が可能でしたら、ぜひ御社で購入したいと考えております。ご検討のほどよろしくお願いいたします。"
            )
        return (
            "お見積もりのご提示ありがとうございます。内容を確認し、前向きに検討しております。\n"
            "念のため確認ですが、お見積もり記載の金額以外に当日追加で発生する費用はございませんでしょうか？\n"
            "また、他社とも比較検討しているのですが、もし日程調整や即決を条件に、端数のお値引きや諸経費のご調整などをご相談できましたら、ぜひ御社にお願いしたいと考えております。ご検討のほどよろしくお願いいたします。"
        )

    # やや割高 / 割高（要交渉） / 危険！要注意！
    if unnecessary_costs:
        names = "・".join(f"「{u.get('item', 'オプション項目')}」" for u in unnecessary_costs[:3])
        saving_sum = sum(int(u.get("potential_saving", 0) or 0) for u in unnecessary_costs)
        saving_note = f"（計{saving_sum:,}円分）" if saving_sum > 0 else ""
        return (
            f"お見積もりのご提示ありがとうございます。内容を拝見し、予算や市場相場と照らし合わせて検討しております。\n"
            f"恐れ入りますが、今回のお見積もりに含まれている{names}{saving_note}につきまして、現在の当方の希望条件では不要と考えておりますため、外していただくことは可能でしょうか？\n"
            f"また、「一式」表記の項目や諸経費の詳しい内訳をご教示いただいた上で、全体の金額を一般的な相場水準まで見直していただけましたら前向きに検討したく存じます。よろしくお願いいたします。"
        )

    return (
        "お見積もりのご提示ありがとうございます。内容を拝見し、他社や市場相場と比較しながら検討しております。\n"
        "大変恐縮ですが、現在の合計金額が当初の予算および一般的な相場よりやや高くなっておりまして、各項目の詳しい内訳をご教示いただくことは可能でしょうか？\n"
        "不要なオプションの整理や日程調整等も含め、もう少しお値引きのご相談が可能でしたら前向きに検討したく存じます。ご確認のほどよろしくお願いいたします。"
    )


def sanitize_and_verify_result(data: dict, category: str = "") -> dict:
    if not isinstance(data, dict):
        return {"status": "error", "message": "無効なデータ形式です。"}
    
    valid_evaluations = [
        "お買い得", "やや割安", "相場通り", "やや割高", "割高（要交渉）", "危険！要注意！", "情報不足（要再確認）"
    ]
    # evaluation の救出（もし status に評価が入っていた場合）
    if "evaluation" not in data or not data["evaluation"]:
        raw_status = str(data.get("status", "")).strip()
        for ve in valid_evaluations:
            if ve in raw_status:
                data["evaluation"] = ve
                break
    data["status"] = "success"

    evaluation = str(data.get("evaluation", "相場通り")).strip()
    if evaluation not in valid_evaluations:
        matched = False
        for ve in valid_evaluations:
            if ve in evaluation:
                evaluation = ve
                matched = True
                break
        if not matched:
            evaluation = "相場通り"
    data["evaluation"] = evaluation

    data["estimated_total_market_price"] = str(data.get("estimated_total_market_price", "") or "")
    data["infrastructure_check"] = str(data.get("infrastructure_check", "") or "")
    data["negotiation_script_line"] = str(data.get("negotiation_script_line", "") or "")
    data["search_keyword"] = str(data.get("search_keyword", "") or "").strip()
    data["recommend_ec_search"] = bool(data.get("recommend_ec_search", False))

    # price_analysis の検証
    pa = data.get("price_analysis")
    if not isinstance(pa, dict):
        pa = {"itemized_list": [], "unnecessary_costs": []}
    
    raw_items = pa.get("itemized_list")
    cleaned_items = []
    positive_total = 0
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                i_name = str(item.get("item", "") or "項目")
                i_price = to_int(item.get("price", 0))
                i_status = str(item.get("status", "相場通り") or "相場通り")
                if i_price > 0:
                    positive_total += i_price
                cleaned_items.append({
                    "item": i_name,
                    "price": i_price,
                    "status": i_status
                })
    pa["itemized_list"] = cleaned_items

    # unnecessary_costs の検証
    raw_unnec = pa.get("unnecessary_costs")
    cleaned_unnec = []
    if isinstance(raw_unnec, list) and evaluation != "情報不足（要再確認）":
        current_saving_total = 0
        for item in raw_unnec:
            if isinstance(item, dict):
                u_name = str(item.get("item", "") or "不要項目")
                u_saving = to_int(item.get("potential_saving", 0))
                u_reason = str(item.get("reason", "") or "")
                if u_saving > 0:
                    # positive_total がある場合、合計が positive_total を超えないよう上限キャップ
                    if positive_total > 0:
                        if current_saving_total + u_saving > positive_total:
                            u_saving = max(0, positive_total - current_saving_total)
                    if u_saving > 0:
                        current_saving_total += u_saving
                        cleaned_unnec.append({
                            "item": u_name,
                            "potential_saving": u_saving,
                            "reason": u_reason
                        })
    pa["unnecessary_costs"] = cleaned_unnec
    data["price_analysis"] = pa

    # 情報不足の場合は安全弁として不要項目と交渉文をクリア
    if evaluation == "情報不足（要再確認）":
        pa["unnecessary_costs"] = []
        data["negotiation_script_line"] = ""
        if not data["estimated_total_market_price"]:
            data["estimated_total_market_price"] = "情報不足のため算出不可"
        if not data["infrastructure_check"]:
            data["infrastructure_check"] = "画像またはテキストから十分な情報が読み取れませんでした。明るい場所で全体がはっきり写るように撮影し直すか、補足情報欄に品名や金額をご入力ください。"
    else:
        raw_script = data["negotiation_script_line"].strip()
        invalid_placeholders = {"なし", "特になし", "不要", "交渉不要", "該当なし", "-"}
        if not raw_script or raw_script in invalid_placeholders or len(raw_script) < 15:
            data["negotiation_script_line"] = build_fallback_negotiation_script(
                evaluation=evaluation,
                category=category,
                unnecessary_costs=cleaned_unnec,
                itemized_list=cleaned_items,
            )
        elif evaluation in ("お買い得", "やや割安", "相場通り"):
            discount_keywords = (
                "値引", "値下", "端数", "割引", "お安く", "カット",
                "上乗せ", "プラス査定", "引き上げ", "サービス", "ポイント", "勉強",
            )
            if not any(k in raw_script for k in discount_keywords):
                if category == "不用品買取":
                    polite_ask = (
                        "また、上記に問題がなければ即決でお願いしたいと考えておりますので、"
                        "もし端数の上乗せやあともう一歩だけ査定額のご配慮をいただくことは可能でしょうか？"
                        "難しければそのままの条件でも前向きに検討いたします。"
                    )
                else:
                    polite_ask = (
                        "また、上記に問題がなければ即決でお願いしたいと考えておりますので、"
                        "もし端数のカットやあともう少しだけお値引き（または付帯サービス等）のご配慮をいただくことは可能でしょうか？"
                        "難しければそのままの金額でも前向きに検討いたします。"
                    )
                # 「このまま即決で〜」のように値引き不要と読める表現があれば置き換え、なければ末尾挨拶の前に挿入
                cleaned_raw = re.sub(
                    r"その上で問題がなければ[^。]*即決で[^。]*。",
                    "",
                    raw_script,
                ).strip()
                closing_match = re.search(r"((?:どうぞ|何卒|ご検討のほど)?よろしくお願(?:い申し上げます|いいたします|いします)。?)\s*$", cleaned_raw)
                if closing_match:
                    prefix = cleaned_raw[: closing_match.start()].rstrip()
                    closing = closing_match.group(1)
                    data["negotiation_script_line"] = f"{prefix}\n{polite_ask}{closing}"
                else:
                    data["negotiation_script_line"] = f"{cleaned_raw}\n{polite_ask}よろしくお願いいたします。"

    return data

_affiliate_cache = {"mtime": 0.0, "data": {}}

def get_affiliate_config() -> dict:
    config_path = os.path.join(BASE_DIR, "affiliate_config.json")
    if not os.path.exists(config_path):
        return {}
    try:
        mtime = os.path.getmtime(config_path)
        if mtime != _affiliate_cache["mtime"] or not _affiliate_cache["data"]:
            with open(config_path, "r", encoding="utf-8") as f:
                _affiliate_cache["data"] = json.load(f)
                _affiliate_cache["mtime"] = mtime
        return _affiliate_cache["data"]
    except Exception as e:
        print(f"Affiliate config load error: {e}")
        return _affiliate_cache.get("data", {})

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(
            content=f.read(),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

@app.get("/icon.png")
@app.get("/app_icon_v2.png")
async def get_icon_v2():
    icon_path = os.path.join(BASE_DIR, "app_icon_v2.png")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    raise HTTPException(status_code=404, detail="Icon not found")

@app.get("/manifest.json")
async def get_manifest():
    manifest_path = os.path.join(BASE_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path)
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/sw.js")
async def get_sw():
    sw_path = os.path.join(BASE_DIR, "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(
            sw_path,
            media_type="application/javascript",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(status_code=404, detail="Service Worker not found")

def resolve_mime_type(filename: Optional[str], content_type: Optional[str]) -> str:
    ct = (content_type or "").strip().lower()
    if ct and ct != "application/octet-stream" and ("/" in ct):
        return ct
    ext = os.path.splitext(filename or "")[1].lower()
    ext_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    return ext_map.get(ext, "image/jpeg")

@app.post("/api/analyze")
async def analyze_images(
    files: Optional[List[UploadFile]] = File(None),
    category: str = Form("エアコン"),
    text_input: Optional[str] = Form(None)
):
    if not GOOGLE_API_KEY or not genai_client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEYが設定されていません。環境変数を設定してください。")
    
    try:
        MAX_FILES = 3
        MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
        image_parts = []
        if files:
            file_list = files if isinstance(files, (list, tuple)) else [files]
            if len(file_list) > MAX_FILES:
                return {"status": "error", "message": f"ファイルは最大{MAX_FILES}点まで選択可能です。"}
            for file in file_list:
                if hasattr(file, "read"):
                    contents = await file.read(MAX_FILE_SIZE + 1)
                    if len(contents) > MAX_FILE_SIZE:
                        return {"status": "error", "message": "1ファイルあたり20MB以下のファイルを選択してください。"}
                    if contents:
                        image_parts.append(
                            types.Part.from_bytes(
                                data=contents,
                                mime_type=resolve_mime_type(
                                    getattr(file, "filename", None),
                                    getattr(file, "content_type", None)
                                ),
                            )
                        )

        if not image_parts and not (text_input and text_input.strip()):
            return {"status": "error", "message": "解析する画像・PDFまたはテキストを入力してください。"}
            
        # カテゴリに応じた固有の指示をプロンプトに追加
        category_instruction = ""
        if category == "車検・整備":
            category_instruction = "【分析対象: 自動車の車検・整備】法定費用（重量税・自賠責・印紙代）は法令で一律のため相場通り。車検基本料（1.5万〜3万円が相場）と推奨整備を厳密に仕分け、不要な各種フラッシング、早期の添加剤、高額消臭コーティング等の過剰オプションを削れる費用として特定してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "家電（冷蔵庫・洗濯機・掃除機など）":
            category_instruction = "【分析対象: 家電（ロボット掃除機・冷蔵庫・洗濯機・テレビ等）】機能・スペックに応じた業界標準の実勢相場を正しく適用してください。\n・ロボット掃除機:\n  - 全自動フラッグシップ（OMNI/全自動ドック、ゴミ収集＋モップ温水洗浄＋熱風乾燥付き等）: 国内実勢相場は『約9万円〜18万円』（定価15万〜25万円クラス）。提示価格が7万〜9万円程度であれば非常に『お買い得』または『やや割安』であり、【絶対に4万〜5万円のエントリー機相場と誤認・混同してはならない】。\n  - ミドルクラス（自動ゴミ収集ステーションのみ、簡易水拭き）: 実勢相場『約4.5万円〜8万円』\n  - エントリークラス（ステーションなし単体機）: 実勢相場『約2万円〜4.5万円』\n・洗濯機: ドラム式洗濯乾燥機（18万〜30万円超）と縦型全自動（4万〜10万円）の相場を明確に区別すること。\n・冷蔵庫: ファミリー向け大容量500L超（18万〜30万円）と中小型2〜3ドア（5万〜12万円）の相場を区別すること。\n※型番が最新・派生型番であっても『中国向け』『国内正規サポート外』等と根拠なく決めつけて危険扱いしないこと。EC最安値比較用のクリーンな 'search_keyword'（メーカー名+シリーズ名/型番）を出力してください。※'recommend_ec_search' は false に設定してください。"
        elif category == "エアコン":
            category_instruction = "【分析対象: エアコン】畳数能力や機能（自動お掃除、換気等）から本体相場を算出。標準工事（配管4m、室内外同階設置、約1.5万〜2万円）と追加工事費（配管延長3,000円〜/m、専用回路増設1.5万〜2.5万円、化粧カバー等）が過剰・重複請求でないか厳格に判定してください。本体型番が特定できる場合はクリーンな 'search_keyword'（メーカー名+型番）を出力してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "ハウスクリーニング":
            category_instruction = "【分析対象: ハウスクリーニング】エアコンクリーニング（通常9,000〜1.4万、お掃除機能付1.5万〜2.2万、室外機3,000〜5,000円）など相場を確認し、過度な防カビコーティング等の不要オプションや不当な出張費をチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "引越し":
            category_instruction = "【分析対象: 引越し】トラックサイズ（単身2t、ファミリー3t〜4t）と時期（通常期/繁忙期）に応じた適正相場を判定し、有料資材や不要な付帯サポートをチェックし、相見積もりの重要性を伝えてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "住宅関連":
            category_instruction = "【分析対象: 住宅リフォーム・外壁塗装・修繕等】足場代（700〜1,000円/㎡）、高圧洗浄（200〜300円/㎡）、外壁塗装（塗料種別に応じた平米単価）を確認。不明瞭な『一式』記載や諸経費10〜15%超の割高請求を厳しく指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "火災保険":
            category_instruction = "【分析対象: 保険全般（火災・地震・生命・医療・自動車保険等）】火災保険の場合は不要な水災特約や家財補償の過剰設定を点検。生命・医療・自動車保険の場合は、家族間での特約重複（個人賠償責任や弁護士特約）や不要な過剰特約を見抜いてください。\n※【重要】火災保険・地震保険の見積もりの場合は、必ず 'recommend_ec_search' を false に設定してください。生命保険・自動車保険・医療保険など火災保険以外の保険見積もりの場合は、必ず 'recommend_ec_search' を true に設定してください。"
        elif category == "ガス料金":
            category_instruction = "【分析対象: プロパンガス（LPガス）】適正相場（基本料金1,500〜2,000円、従量単価300〜500円/㎥）と比較し、従量単価600円以上の高額請求を厳重に指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "電気料金":
            category_instruction = "【分析対象: 電気料金】基本料金や電力量料金単価、および燃料費調整額の上限撤廃や市場連動型プランのリスクを確認してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "パソコン・スマホ購入・修理":
            category_instruction = "【分析対象: スマホ・PC・周辺機器】CPU/SoC世代（Apple Mシリーズ、Intel Core Ultra、Snapdragon等）、メモリ容量、SSD容量、画面仕様から市場相場を精密に推計。\n・折りたたみ最上位フラッグシップ（Galaxy Z Foldシリーズ等）: 国内実勢相場は『約24万円〜32万円』（大容量モデル含む）。過去の旧型番の型落ち相場に引き下げて割高判定する旧世代ダウングレード比較を絶対に禁止すること。最新・次世代型番であっても『未発表』『未発売』と決めつけず、最高峰フラッグシップ相場を正当に適用し、25万円前後は『相場通り』または『お買い得』と正当評価せよ。\n・MacBook Proやハイエンド機にも型落ちローエンド相場を誤認適用しないこと。\n・新品購入時の不要な高額初期設定サポート（1万〜3万円）や不要オプションを見抜くこと。スマートフォン本体、PC本体、タブレット、周辺機器等の場合は最安値比較用のクリーンな 'search_keyword'（メーカー名+型番/機種名）を設定してください。※'recommend_ec_search' は false に設定してください。"
        elif category == "不用品買取":
            category_instruction = "【分析対象: 不用品買取】出張費・査定料・キャンセル料の無料確認を行い、貴金属やブランド品の不当な安値買い叩き（押し買い）や違法回収費用がないかチェックしてください。提示された査定額が市場相場より高ければ『お買い得』等の好条件判定とし、不当に安く買い叩かれている場合は『割高（要交渉）』や『危険！要注意！』と判定してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        else:
            category_instruction = f"【分析対象: {category}】型番や名称の直接一致がない場合でも、提示された機能・仕様・構成・役務内容から論理的に同等水準の国内適正相場を客観的に割り出し、価格の妥当性を監査してください。決して『国内未発表』『中国向け』などと推測で決めつけず、同等スペックの日本市場相場を明示してください。有形商品であれば 'search_keyword' に最安値比較用の的確なキーワードを設定してください。"

        if text_input:
            prompt = f"以下の見積もりテキスト、および（添付があれば）画像を解析し、指定されたJSON形式で分析結果を出力してください。\n\n【見積もりテキスト】\n{text_input}\n\n{category_instruction}"
        else:
            prompt = f"添付された見積書や設置環境の画像を確認し、以下の指示に従って指定されたJSON形式で分析結果を出力してください。\n\n{category_instruction}"
        
        response, last_error = await generate_with_smart_fallback([prompt] + image_parts)

        if not response or not response.text:
            return {"status": "error", "message": f"AIモデルの解析に失敗しました。エラー詳細: {last_error}"}
            
        raw_json = None
        clean_text = response.text.strip()
        # ```json ... ``` があればそれを最優先抽出
        json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_text)
        candidate = json_block_match.group(1).strip() if json_block_match else clean_text

        try:
            raw_json = json.loads(candidate, strict=False)
        except Exception:
            brace_match = re.search(r'(\{[\s\S]*\})', candidate)
            if brace_match:
                try:
                    raw_json = json.loads(brace_match.group(1), strict=False)
                except Exception as parse_err:
                    print(f"JSON Parse Error: {parse_err}")
            
        if not raw_json or not isinstance(raw_json, dict):
            return {"status": "error", "message": "AIからの応答を正しく解析できませんでした。もう一度お試しください。"}
        
        # Python自動検算・サニタイズ処理
        result_json = sanitize_and_verify_result(raw_json, category=category)
        if result_json.get("status") == "error":
            return result_json

        # === Step 3: アフィリエイトリンクの動的付与 ===
        try:
            affiliate_config = get_affiliate_config()
            if category in affiliate_config:
                recommend_ec = result_json.get("recommend_ec_search", False)
                
                if not recommend_ec:
                    eval_text = result_json.get("evaluation", "")
                    is_high_priority = any(keyword in eval_text for keyword in ["危険", "割高", "要注意"])
                    
                    result_json["affiliate_recommendation"] = {
                        "is_active": True,
                        "is_high_priority": is_high_priority,
                        "title": affiliate_config[category]["title"],
                        "name": affiliate_config[category]["name"],
                        "url": affiliate_config[category]["url"],
                        "description": affiliate_config[category]["description"]
                    }
        except Exception as e:
            print(f"Affiliate config error: {e}")
            pass

        return result_json
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
