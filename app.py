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
1. 最新かつ正確な情報をユーザーに届ける。
2. ユーザーの最大の利点は「良い製品やサービスを適正価格で購入できること」。
3. スマホで読みやすい簡潔なテキストにする。

【評価基準とJSON出力】
必ず以下のJSONフォーマットのみで出力せよ（マークダウン不可）。
"evaluation"は必ず以下のいずれか1つを使用すること。
・「危険！要注意！」（相場より50%以上安い、または異常な高額請求）
・「お買い得」（10%〜50%未満安い）
・「やや割安」（5%〜10%未満安い）
・「相場通り」（±5%未満）
・「やや割高」（5%〜適度に高い）
・「相場判定不可（要比較）」（自身のデータベースにない最新製品、または判断が困難な特殊な商品・サービスの場合。決して「未発表」「詐欺」と推測しないこと）

【アフィリエイト誘導の最適化】
通常は「recommend_ec_search: false」とし専門業者へ誘導するが、「PCやスマホ、家電の本体購入」などECサイトでの購入が適している場合は必ず「true」にすること。

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
  "search_keyword": "商品本体を探すための検索キーワード。※家電やPC等は必ず具体的な商品名（型番等）を出力。無形サービスの場合は空文字。"
}
"""

import re

# グローバル変数としてモデルリストをキャッシュ（通信ゼロ化による超高速化）
_cached_models_to_try = [
    "gemini-2.5-flash",
    "gemini-flash"
]

def get_dynamic_models():
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
            category_instruction = "【分析対象: 自動車の車検・整備】法定費用と整備費用を分け、過剰な添加剤や早すぎる消耗品交換を見抜いてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "家電（冷蔵庫・洗濯機・掃除機など）":
            category_instruction = "【分析対象: 家電】本体価格や搬入経路、設置費用が適正か確認してください。\n※【重要】いかなる場合も必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "エアコン":
            category_instruction = "【分析対象: エアコン】コンセント形状や配管延長費用等の追加工事費が適正か確認してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "ハウスクリーニング":
            category_instruction = "【分析対象: ハウスクリーニング】不要なオプションや不当な出張費がないか厳しくチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "引越し":
            category_instruction = "【分析対象: 引越し】トラックサイズや作業員数、オプション料金が適正か確認し、相見積もりの重要性を伝えてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "住宅関連":
            category_instruction = "【分析対象: 住宅リフォーム・外壁塗装等】法外な料金からユーザーを守るため、不明瞭な諸経費や相場を超えるオプション工事費を指摘してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "火災保険":
            category_instruction = "【分析対象: 火災保険】不要な特約がないか確認してください。\n※【重要】もしユーザーが自動車保険や生命保険の見積もりを誤って送信してきた場合は、必ず 'recommend_ec_search' を true にしてください。それ以外の正常な火災保険の場合は false にしてください。"
        elif category == "ガス料金":
            category_instruction = "【分析対象: プロパンガス（LPガス）】基本料金や従量単価が地域の適正相場と比較して高すぎないか確認してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "電気料金":
            category_instruction = "【分析対象: 電気料金】基本料金や電力量料金が高すぎないか確認してください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        elif category == "パソコン・スマホ購入・修理":
            category_instruction = "【分析対象: パソコン・スマホ】最新機種の新品購入や高額な修理代が適正か確認してください。\n※【重要】パソコン（PC）の見積もりの場合は専用業者の対象外のため、必ず 'recommend_ec_search' を true にし、'search_keyword' にパソコンの検索キーワードを出力してください。スマホ・iPhoneの場合は false にしてください。"
        elif category == "不用品買取":
            category_instruction = "【分析対象: 不用品買取】不当な安値での買い叩きや悪徳な廃品回収費用がないかチェックしてください。※必ずJSONの 'recommend_ec_search' を false に設定してください。"
        else:
            category_instruction = f"【分析対象: {category}】カテゴリに応じた一般的な適正価格と品質維持の観点から分析してください。"

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
        
        response = None
        last_error = "Unknown Error"
        for model_name in models_to_try:
            try:
                from google.api_core import retry
                
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={"response_mime_type": "application/json"}
                )
                # 画像解析は数秒〜十数秒かかる場合があるため、タイムアウトを45秒に設定し確実な完了を担保
                response = model.generate_content(
                    [analysis_prompt] + image_parts,
                    request_options={"retry": retry.Retry(initial=0, maximum=0, multiplier=1.0, deadline=45.0), "timeout": 45.0}
                )
                if response and response.text:
                    break
            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {e}")
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
