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

SYSTEM_INSTRUCTION = """
【厳守事項】
1. 最新かつ正確な情報をユーザーに届ける。ハルシネーション（推測による事実無根の断定）を徹底排除する。
2. 見積もりの各項目を専門鑑定士の視点で精査し、「相場通りの正当な費用」と「不要・過剰請求・相場超えの費用」を厳密に仕分ける。
3. ユーザーがそのまま使える論理的かつ角の立たない交渉アドバイスを提供する。
4. スマホで即座に理解できるよう、冗長な前置きや重複を排し、簡潔・論理的・高品質なテキストを出力する。

【金額・計算ルールの徹底】
・見積書の税抜・税込表記を確認し、各項目の金額（price）と削減見込み額（potential_saving）は必ず整数の数値型（number）で出力せよ（カンマや記号不可）。
・不透明な「諸経費」「一式」の記載には内訳開示を促すこと。

【評価基準とJSON出力】
必ず以下のJSONフォーマットのみで出力せよ（マークダウン装飾、前置きテキスト不可）。
"evaluation"は必ず以下のいずれか1つを使用すること。
・「危険！要注意！」（相場より50%以上安い・高い、または不当な過剰請求・不審点がある）
・「お買い得」（10%〜50%未満安い）
・「やや割安」（5%〜10%未満安い）
・「相場通り」（±5%未満）
・「やや割高」（5%〜適度に高い）
・「相場判定不可（要比較）」（判断が困難な特殊な商品・サービスの場合。決して「未発表」「詐欺」と推測しないこと）

【EC検索キーワード（search_keyword）の厳格ルール】
・対象が見積もり対象の有形商品（家電、スマホ、PC、機器、パーツ等）の場合、楽天市場やYahoo!ショッピングで最安値ショップをドンピシャで比較できるよう、必ず「メーカー名 型番・モデル名（例: パナソニック NA-LX129CL、ダイキン S223ATES、Apple iPhone 15 128GB）」をノイズなしでクリーンに出力せよ。
・無形サービス（引越し、ハウスクリーニング、保険、ガス等の作業費中心）で特定の商品がない場合は、必ず空文字 "" を出力せよ。

{
  "status": "success",
  "evaluation": "上記のいずれか",
  "estimated_total_market_price": "見積もり全体の適正相場（例: '約15万円〜20万円'）※不明な場合は『判定不可』",
  "infrastructure_check": "スマホで読めるインフラ適合確認結果と注意事項",
  "price_analysis": {
    "itemized_list": [ {"item": "項目名", "price": 0, "status": "相場通り/割高"} ],
    "unnecessary_costs": [ {"item": "項目名", "potential_saving": 0, "reason": "理由"} ]
  },
  "negotiation_script_line": "LINE用の値引き・内訳開示交渉スクリプト",
  "negotiation_script_shop": ["店頭用カンペ1", "店頭用カンペ2"],
  "recommend_ec_search": false,
  "search_keyword": "商品本体を探すためのクリーンな検索キーワード（メーカー名+型番等）。無形サービスは空文字。"
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
        
    base_models = []
    try:
        # 常にGoogleのサーバーから最新のモデル一覧を全自動取得（固有名詞のハードコード完全排除）
        available = [
            m.name.replace('models/', '') 
            for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods and 'flash' in m.name
        ]
        
        # プレビュー版や特殊用途（画像生成特化、音声特化、実験版等）を除外し、安定した見積もり解析モデルのみを抽出
        excluded_keywords = [
            "preview", "eap", "lite", "omni", "image", "tts", 
            "audio", "native", "transcribe", "computer-use", "robotics"
        ]
        filtered = [
            m for m in available 
            if not any(k in m for k in excluded_keywords)
        ]
        
        # バージョン番号で降順ソート（常に最新モデルが1位、準最新が2位、第3位…となる）
        filtered.sort(key=lambda x: extract_version(x), reverse=True)
        
        for m in filtered:
            if m not in base_models:
                base_models.append(m)
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        pass
        
    # 取得結果をキャッシュ。上位5つの最新〜準最新モデルを順次フォールバック用として保持
    if base_models:
        _cached_models_to_try = base_models[:5]
    else:
        # 万が一Googleの一覧取得API自体が一時通信遮断で失敗した場合も、固有名詞を一切使わず
        # Google公式の動的最新エイリアスを緊急フォールバックとして使用
        _cached_models_to_try = ["gemini-flash-latest", "gemini-flash"]
        
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
            category_instruction = "【分析対象: 家電（冷蔵庫・洗濯機・掃除機等）】本体価格がEC等の実勢相場と比較して適正か、リサイクル回収運搬費や特殊設置費用が過剰でないか確認してください。\n※【重要】いかなる場合も必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "エアコン":
            category_instruction = "【分析対象: エアコン】標準工事（配管4m、室内外同階設置）は約1.5万〜2万円が相場。配管延長（3,000円〜/m）、専用コンセント増設（1.5万〜2.5万円）、化粧カバー（6,000円〜）、室外機特殊設置等の追加工事費が過剰・重複請求でないか厳格に判定してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "ハウスクリーニング":
            category_instruction = "【分析対象: ハウスクリーニング】エアコンクリーニング（通常9,000〜1.4万、お掃除機能付1.5万〜2.2万、室外機3,000〜5,000円）など相場を確認し、過度な防カビコーティング等の不要オプションや不当な出張費をチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "引越し":
            category_instruction = "【分析対象: 引越し】トラックサイズ（単身2t、ファミリー3t〜4t）と時期（通常期/繁忙期）に応じた適正相場を判定し、有料資材や不要な付帯サポートをチェックし、相見積もりの重要性を伝えてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "住宅関連":
            category_instruction = "【分析対象: 住宅リフォーム・外壁塗装・修繕等】足場代（700〜1,000円/㎡）、高圧洗浄（200〜300円/㎡）、外壁3回塗りの確認を行い、不明瞭な『一式』記載や諸経費10〜15%超の割高請求を厳しく指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "火災保険":
            category_instruction = "【分析対象: 保険全般（火災・地震・生命・医療・自動車保険等）】火災保険の場合は不要な水災特約や家財補償の過剰設定を点検。生命・医療・自動車保険の場合は、家族間での特約重複（個人賠償責任や弁護士特約）や不要な過剰特約を見抜いてください。\n※【重要】火災保険以外の生命保険・自動車保険・医療保険の場合は、必ず 'recommend_ec_search' を true にしてください。正常な火災保険の場合は false にしてください。"
        elif category == "ガス料金":
            category_instruction = "【分析対象: プロパンガス（LPガス）】適正相場（基本料金1,500〜2,000円、従量単価300〜500円/㎥）と比較し、従量単価600円以上の高額請求を厳重に指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "電気料金":
            category_instruction = "【分析対象: 電気料金】基本料金や電力量料金単価、および燃料費調整額の上限撤廃や市場連動型プランのリスクを確認してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "パソコン・スマホ購入・修理":
            category_instruction = "【分析対象: スマホ・PC・周辺機器】実勢価格や修理代の妥当性を確認し、新品購入時の不要な初期設定サポート（1万〜3万円）や高額な付帯オプションを見抜いてください。\n※【重要】パソコン本体や周辺機器の見積もりの場合は、必ず 'recommend_ec_search' を true にし、'search_keyword' に商品名・型番を出力してください。スマホ・iPhone本体・修理の場合は false にしてください。"
        elif category == "不用品買取":
            category_instruction = "【分析対象: 不用品買取】出張費・査定料・キャンセル料の無料確認を行い、貴金属やブランド品の不当な安値買い叩き（押し買い）や違法回収費用がないかチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        else:
            category_instruction = f"【分析対象: {category}】一般的な市場相場と品質維持の観点から適正価格か精査してください。"

        if text_input:
            prompt = f"以下の見積もりテキスト、および（添付があれば）画像を解析し、指定されたJSON形式で分析結果を出力してください。\n\n【見積もりテキスト】\n{text_input}\n\n{category_instruction}"
        else:
            prompt = f"添付された見積書や設置環境の画像を確認し、以下の指示に従って指定されたJSON形式で分析結果を出力してください。\n\n{category_instruction}"
        
        # 翼さんのアイデアを完璧な形で実装：
        # 遅延の原因だった「毎回の一覧取得」を初回1回のみ（キャッシュ化）にすることで待機時間をゼロにしつつ、
        # 3.7や3.6などの固有名詞を一切ハードコーディングせず、常にGoogleの最新無料モデルを動的に取得・フォールバックする。
        models_to_try = get_dynamic_models()
        
        # === 1ステップで全解析を完了（処理時間を大幅削減） ===
        # SDKエラーと遅延の元凶であったWeb検索機能への依存を廃止し、最新モデルの内部知識と厳格なハルシネーション対策プロンプトで精度と速度を両立。
        
        analysis_prompt = f"【注意事項】\n対象製品の存在や適正相場について不確実な点がある場合は、完全に断定せず、確認をおすすめするアドバイスにとどめてください。\n\n{prompt}"
        
        # 上限超過（429）となったモデルのチェック（1時間経過したものは自動回復させて再挑戦）
        current_time = time.time()
        active_exhausted = {
            m for m, t in _exhausted_models.items() if current_time - t < 3600
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
                        "max_output_tokens": 2048
                    }
                )
                # タイムアウト15秒。リトライで粘らず、上限エラーや不達時は「0.1秒」で即座に次のモデルへフォールバック
                response = model.generate_content(
                    [analysis_prompt] + image_parts,
                    request_options={"timeout": 15.0}
                )
                if response and response.text:
                    break
            except Exception as e:
                last_error = str(e)
                err_str = str(e)
                print(f"Model {model_name} failed: {e}")
                # 429（上限超過）が発生したモデルは記録し、次のユーザーからは最初から生きているモデルで即起動
                if "429" in err_str or "ResourceExhausted" in err_str:
                    _exhausted_models[model_name] = time.time()
                continue
                
        if not response or not response.text:
            return {"status": "error", "message": f"AIモデルの解析に失敗しました。エラー詳細: {last_error}"}
            
        try:
            # 応答テキストをJSONとしてパース（Markdownの余分な装飾を剥がす）
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            result_json = json.loads(clean_text)
        except json.JSONDecodeError:
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
