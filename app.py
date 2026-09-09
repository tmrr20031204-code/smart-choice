import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import google.generativeai as genai
from typing import List, Optional

app = FastAPI()

# 既存プロジェクトとの隔離のため、環境変数から取得します。
# .envファイルがあれば自動で読み込む処理（全自動化対応）
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k] = v

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_INSTRUCTION = """あなたは、見積書・価格・プランを極めて客観的かつ適正に評価する独立系・最高峰の価格監査・相場判定AI「SmartChoice」です。

【判定放棄・思考停止・逃げの絶対禁止（最重要ルール）】
・「国内未発表」「未発売」「型番不明のため解析不能」「情報がありません」「判定不可」といった理由で、解析・相場推計・適正評価を放棄・回答回避することは【絶対に禁止】です。
・家電、ガジェット、家具、自動車関連、住宅リフォーム、各種役務サービスなど、あらゆる対象において、型番が完全に特定できない場合や海外モデル・最新型であっても、提示された画像やテキストから読み取れる【スペック・機能・仕様・部材構成・作業工程・提供規模】をプロの専門アナリストとして論理的に分解・精査してください。
・同等クラスの性能・品質を持つ日本国内市場の実勢相場を客観的に導き出し、「見積もり全体の適正相場（下限〜上限）」を【必ず具体的に算出・明示】してください。

【プロの価格監査原則】
1. 最新かつ正確な市場知識を論理的に駆使し、ハルシネーション（根拠のないデタラメ）を徹底排除する。
2. 見積もりの各項目を専門鑑定士の視点で精査し、「相場通りの正当な費用」と「不要・過剰請求・相場超えの費用」を厳密に仕分ける。
3. ユーザーが不利益を被らないよう、曖昧な「一式」「諸経費」には内訳開示を促し、具体的かつ角の立たない交渉アドバイスを提供する。
4. スマホで即座に理解できるよう、冗長な前置きや重複を排し、簡潔・論理的・最高品質なテキストを出力する。

【金額・計算ルールの徹底】
・見積書の税抜・税込表記を確認し、各項目の金額（price）と削減見込み額（potential_saving）は必ず整数の数値型（number）で出力せよ（カンマや記号不可）。

【評価基準（evaluation）】
必ず以下のいずれか1つを厳密に選定せよ。
・「危険！要注意！」（相場より大幅に高い/安い、または過剰請求・不審点がある場合）
・「お買い得」（相場より10%〜50%程度割安で品質・機能が十分な場合）
・「やや割安」（相場より5%〜10%程度安い場合）
・「相場通り」（適正相場範囲内（±5%以内）の場合）
・「やや割高」（相場より5%〜20%程度高い場合）

【EC検索キーワード（search_keyword）の厳格ルール】
・対象が有形商品（家電、ガジェット、家具、PC、スマホ、部品等）の場合、楽天市場やYahoo!ショッピングで最安値比較ができるよう、最も的確な「メーカー名 型番」または「ジャンル 主要スペック（例: ロボット掃除機 自動ゴミ収集 水拭き）」をノイズなしでクリーンに出力せよ。
・無形サービス（引越し、リフォーム、保険、ガス等の作業費中心）で特定の商品がない場合は、空文字 "" を出力せよ。

【出力仕様】
必ず以下のJSONフォーマットのみで出力せよ（マークダウン装飾、前置きテキスト不可）。
{
  "status": "success",
  "evaluation": "上記の評価基準のいずれか1つ",
  "estimated_total_market_price": "見積もり全体の適正相場（例: '約15万円〜22万円'）。決して判定不可と逃げず必ず相場範囲を明示すること",
  "infrastructure_check": "スペック・機能の分析、同等モデルとの市場相場比較、購入・契約時の注意点やプロのアドバイス（簡潔かつ具体的・専門的に）",
  "price_analysis": {
    "itemized_list": [ {"item": "項目名", "price": 0, "status": "相場通り/割高/割安"} ],
    "unnecessary_costs": [ {"item": "項目名", "potential_saving": 0, "reason": "理由"} ]
  },
  "negotiation_script_line": "LINE用の値引き・内訳開示交渉スクリプト",
  "negotiation_script_shop": ["店頭用カンペ1", "店頭用カンペ2"],
  "recommend_ec_search": false,
  "search_keyword": "商品本体を探すためのクリーンな検索キーワード。無形サービスは空文字。"
}
"""

import re
import time

# グローバル変数としてモデルリストおよび上限超過モデルをキャッシュ
_cached_models_to_try = []
_exhausted_models = {}  # {model_name: timestamp_when_exhausted}

def extract_version(name):
    # gemini-3.8-flash や将来の gemini-4-flash / gemini-4.5-flash などのバージョン番号を正確に数値化
    match = re.search(r'gemini-(\d+(?:\.\d+)?)-flash', name)
    if match:
        return float(match.group(1))
    return 0.0

def get_dynamic_models():
    global _cached_models_to_try
    if _cached_models_to_try:
        return _cached_models_to_try
        
    primary_models = []
    fallback_models = []
    try:
        # 常にGoogleのサーバーから最新のモデル一覧を全自動取得（固有名詞のハードコード完全排除）
        available = [
            m.name.replace('models/', '') 
            for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name
        ]
        
        # 特殊用途（画像生成特化、音声特化、実験版、および停止中の旧世代）を除外
        excluded_keywords = [
            "preview", "eap", "omni", "image", "tts", 
            "audio", "native", "transcribe", "computer-use", "robotics",
            "3.5-flash", "3.6-flash"  # 応答停止中の旧世代通常Flashは除外
        ]
        filtered = [
            m for m in available 
            if not any(k in m for k in excluded_keywords)
        ]
        
        # 通常の最新Flashモデル（最高精度重視）
        standard_flash = [m for m in filtered if "lite" not in m]
        standard_flash.sort(key=lambda x: extract_version(x), reverse=True)
        
        # 高クォータ・超高速の最新Liteモデル（20回上限到達時の無尽蔵フォールバック）
        lite_flash = [m for m in filtered if "lite" in m]
        lite_flash.sort(key=lambda x: extract_version(x), reverse=True)
        
        # 公式の動的エイリアスも含め、最適な試行順序で構成
        primary_models = standard_flash
        fallback_models = lite_flash
        if "gemini-flash-lite-latest" not in fallback_models:
            fallback_models.insert(0, "gemini-flash-lite-latest")
            
        combined = primary_models + fallback_models
        for m in combined:
            if m not in _cached_models_to_try:
                _cached_models_to_try.append(m)
                
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        pass
        
    if not _cached_models_to_try:
        _cached_models_to_try = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-flash"]
        
    return _cached_models_to_try

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/icon.png")
async def get_icon():
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    raise HTTPException(status_code=404, detail="Icon not found")

@app.get("/app_icon_v2.png")
async def get_icon_v2():
    icon_path = os.path.join(os.path.dirname(__file__), "app_icon_v2.png")
    if os.path.exists(icon_path):
        return FileResponse(icon_path)
    raise HTTPException(status_code=404, detail="Icon not found")

@app.get("/manifest.json")
async def get_manifest():
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path)
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/sw.js")
async def get_sw():
    sw_path = os.path.join(os.path.dirname(__file__), "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service Worker not found")

@app.post("/api/analyze")
async def analyze_images(
    files: Optional[List[UploadFile]] = File(None),
    category: str = Form("エアコン"),
    text_input: Optional[str] = Form(None)
):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEYが設定されていません。環境変数を設定してください。")
    
    try:
        image_parts = []
        if files:
            for file in files:
                contents = await file.read()
                if contents:
                    image_parts.append(
                        {
                            "mime_type": file.content_type,
                            "data": contents
                        }
                    )
            
        # カテゴリに応じた固有の指示をプロンプトに追加
        category_instruction = ""
        if category == "車検・整備":
            category_instruction = "【分析対象: 自動車の車検・整備】法定費用（重量税・自賠責・印紙代）は法令で一律のため相場通り。車検基本料（1.5万〜3万円が相場）と推奨整備を厳密に仕分け、不要な各種フラッシング、早期の添加剤、高額消臭コーティング等の過剰オプションを削れる費用として特定してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "家電（冷蔵庫・洗濯機・掃除機など）":
            category_instruction = "【分析対象: 家電（冷蔵庫・洗濯機・掃除機等）】ロボット掃除機、洗濯機、冷蔵庫、テレビ、季節家電等。型番が完全一致しない場合や未記載でも、LiDAR・水拭き・自動ゴミ収集・容量・インバーター等の【機能・スペック・グレード】から同等の国内市場実勢相場（適正相場範囲）を論理的に算出し、決して『未発表』『判定不可』と逃げないこと。本体価格の妥当性、不要な長期延長保証や高額な設置配送費の有無を監査してください。EC最安値比較用のクリーンな 'search_keyword'（例: 'ロボット掃除機 自動ゴミ収集 水拭き' やメーカー名型番）を出力してください。※'recommend_ec_search' は false に設定してください（提携EC枠を優先活用します）。"
        elif category == "エアコン":
            category_instruction = "【分析対象: エアコン】畳数能力や機能（自動お掃除、換気等）から本体相場を算出。標準工事（配管4m、室内外同階設置、約1.5万〜2万円）と追加工事費（配管延長3,000円〜/m、専用回路増設1.5万〜2.5万円、化粧カバー等）が過剰・重複請求でないか厳格に判定してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "ハウスクリーニング":
            category_instruction = "【分析対象: ハウスクリーニング】エアコンクリーニング（通常9,000〜1.4万、お掃除機能付1.5万〜2.2万、室外機3,000〜5,000円）など相場を確認し、過度な防カビコーティング等の不要オプションや不当な出張費をチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "引越し":
            category_instruction = "【分析対象: 引越し】トラックサイズ（単身2t、ファミリー3t〜4t）と時期（通常期/繁忙期）に応じた適正相場を判定し、有料資材や不要な付帯サポートをチェックし、相見積もりの重要性を伝えてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "住宅関連":
            category_instruction = "【分析対象: 住宅リフォーム・外壁塗装・修繕等】足場代（700〜1,000円/㎡）、高圧洗浄（200〜300円/㎡）、外壁塗装（塗料種別に応じた平米単価）を確認。不明瞭な『一式』記載や諸経費10〜15%超の割高請求を厳しく指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "火災保険":
            category_instruction = "【分析対象: 保険全般（火災・地震・生命・医療・自動車保険等）】火災保険の場合は不要な水災特約や家財補償の過剰設定を点検。生命・医療・自動車保険の場合は、家族間での特約重複（個人賠償責任や弁護士特約）や不要な過剰特約を見抜いてください。\n※【重要】火災保険以外の生命保険・自動車保険・医療保険の場合は、必ず 'recommend_ec_search' を true にしてください。正常な火災保険の場合は false にしてください。"
        elif category == "ガス料金":
            category_instruction = "【分析対象: プロパンガス（LPガス）】適正相場（基本料金1,500〜2,000円、従量単価300〜500円/㎥）と比較し、従量単価600円以上の高額請求を厳重に指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "電気料金":
            category_instruction = "【分析対象: 電気料金】基本料金や電力量料金単価、および燃料費調整額の上限撤廃や市場連動型プランのリスクを確認してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "パソコン・スマホ購入・修理":
            category_instruction = "【分析対象: スマホ・PC・周辺機器】CPU・メモリ・SSD容量、画面仕様、修理内容から市場相場を推計。新品購入時の不要な高額初期設定サポート（1万〜3万円）や不要オプションを見抜いてください。パソコン本体や周辺機器の見積もりの場合は、クリーンな 'search_keyword' を設定してください。※'recommend_ec_search' は false に設定してください。"
        elif category == "不用品買取":
            category_instruction = "【分析対象: 不用品買取】出張費・査定料・キャンセル料の無料確認を行い、貴金属やブランド品の不当な安値買い叩き（押し買い）や違法回収費用がないかチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        else:
            category_instruction = f"【分析対象: {category}】型番や名称の直接一致がない場合でも、機能・仕様・構成・役務内容から論理的に同等水準の適正相場を割り出し、客観的かつ厳格に価格の妥当性を監査してください。有形商品であれば 'search_keyword' に最安値比較用の的確なキーワードを設定してください。"

        if text_input:
            prompt = f"以下の見積もりテキスト、および（添付があれば）画像を解析し、指定されたJSON形式で分析結果を出力してください。\n\n【見積もりテキスト】\n{text_input}\n\n{category_instruction}"
        else:
            prompt = f"添付された見積書や設置環境の画像を確認し、以下の指示に従って指定されたJSON形式で分析結果を出力してください。\n\n{category_instruction}"
        
        # 翼さんのアイデアを完璧な形で実装：
        # 遅延の原因だった「毎回の一覧取得」を初回1回のみ（キャッシュ化）にすることで待機時間をゼロにしつつ、
        # 3.7や3.6などの固有名詞を一切ハードコーディングせず、常にGoogleの最新無料モデルを動的に取得・フォールバックする。
        models_to_try = get_dynamic_models()
        
        # === 1ステップで全解析を完了（超高速かつ高精度） ===
        analysis_prompt = prompt
        
        # 上限超過やエラーとなったモデルのチェック（3分経過したものは自動回復させて再挑戦）
        current_time = time.time()
        active_exhausted = {
            m for m, t in _exhausted_models.items() if current_time - t < 180
        }
        
        # 上限超過モデルを一時的にリストの末尾に回し、今すぐ動くモデルを最優先にして待ち時間をゼロ化
        prioritized_models = [m for m in models_to_try if m not in active_exhausted] + [m for m in models_to_try if m in active_exhausted]
        
        response = None
        last_error = "Unknown Error"
        for model_name in prioritized_models:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                        "max_output_tokens": 4096
                    }
                )
                # タイムアウト12秒。リトライで粘らず、上限エラーや不達時は即座に次のモデルへ高速フォールバック
                response = model.generate_content(
                    [analysis_prompt] + image_parts,
                    request_options={"timeout": 12.0}
                )
                if response and response.text:
                    break
            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {e}")
                # 429（上限超過）や504（タイムアウト）など、応答不能なモデルは即座に除外キャッシュに記録
                # 次のユーザーからは生きている高速モデルが最優先で直結され、待機時間を極限まで削減
                _exhausted_models[model_name] = time.time()
                continue
                
        if not response or not response.text:
            return {"status": "error", "message": f"AIモデルの解析に失敗しました。エラー詳細: {last_error}"}
            
        try:
            # 応答テキストをJSONとしてパース（Markdownの余分な装飾を剥がす）
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(clean_text)
        except json.JSONDecodeError:
            # 前後に余計なテキストがある場合のフォールバック正規表現抽出
            json_match = re.search(r'(\{[\s\S]*\})', clean_text)
            if json_match:
                try:
                    result_json = json.loads(json_match.group(1))
                except Exception:
                    return {"status": "error", "message": "AIからの応答を正しく解析できませんでした。"}
            else:
                return {"status": "error", "message": "AIからの応答を正しく解析できませんでした。"}
        
        # === Step 3: アフィリエイトリンクの動的付与 ===
        try:
            config_path = os.path.join(os.path.dirname(__file__), "affiliate_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    affiliate_config = json.load(f)
                
                if category in affiliate_config:
                    # AIがECサイトへのフォールバックを強く推奨しているか確認
                    recommend_ec = result_json.get("recommend_ec_search", False)
                    
                    if not recommend_ec:
                        eval_text = result_json.get("evaluation", "")
                        # 悪質判定かどうかのフラグ
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
            # アフィリエイト読み込みエラーはメイン処理に影響させない
            print(f"Affiliate config error: {e}")
            pass

        return result_json
        
    except Exception as e:
        # 予期せぬエラー用
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    # Render等のクラウド環境では環境変数 PORT にポート番号が渡される
    port = int(os.environ.get("PORT", 8000))
    # python app.py で実行可能
    uvicorn.run(app, host="0.0.0.0", port=port)
