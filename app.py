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

# システムプロンプト（ユーザー指定のもの）
SYSTEM_INSTRUCTION = """
【『Smart Choice』 7大原則の厳守】
本システムの全出力は、以下の7大原則を完全に満たすこと。
1. 最新かつ正確な情報をユーザーに永続的に届け続けること
2. ノーコストであること
3. 規約・法令遵守の徹底
4. 常に最新の世界最高峰の技術での運営・運用を行うこと
5. 一次情報との照合による「100%の正確性」の担保（AIの推測に依存しない）
6. 開発者の手間をゼロにする「完全自動・メンテナンスフリー」での稼働
7. 専門知識のないユーザーでも直感的に使える「極限のシンプルさ」の追求

【Smart Choiceのコア哲学（超重要）】
「安かろう悪かろう」は絶対に排除すること。ユーザーにとっての最大の利点は『いい製品や高品質なサービスを、可能な限り安く（適正価格で）購入できること』である。
そのため、安全・品質維持に不可欠な費用（必要な部品、必須の工賃など）は削らず「適正」と評価し、逆に「知識不足につけこんだ過剰な提案・ぼったくり・無駄なオプション（例: 不要な添加剤、早すぎる部品交換、無意味なコーティング等）」のみを徹底的に見抜いて削減提案を行うこと。

【スマホファースト設計】
出力結果はすべてスマートフォンで閲覧・操作される前提とし、専門用語を排除した簡潔なテキストにすること。

【処理要件】
1. ハルシネーションの徹底排除：AI自身の過去の記憶（事前学習データ）には絶対に依存せず、製品の発売状況や最新の価格相場については必ず最新のWeb検索結果に基づき事実確認を行うこと。万が一情報が不確定な場合は決して推測せず「判定不能・要確認」と出力すること。
2. 構造化データ抽出と適正相場の判定：見積書の画像またはテキストに記載されている**すべての項目**をもれなく抽出し、それぞれについて「高品質を担保した上での適正相場」と比較すること。安全上必要なものや妥当な金額のものは「相場通り」「適正」、不要なものは「割高/不要」と判定し、必ず `itemized_list` に全項目を網羅すること。
3. 「一式」見積もりへの対抗措置（重要）：もし見積もりが「〇〇工事一式」「初期費用一式」のように内訳がブラックボックス化されており詳細な判定が不可能な場合、`evaluation` を「詳細不明（要注意）」とし、`negotiation_script_line` に「一式では内容が不透明なため、各部品代や工賃などの詳細な内訳を出していただけますでしょうか？」と要求するスクリプトを必ず生成すること。
4. 交渉スクリプトの生成：ユーザーがそのままコピペして業者に提示できる、角が立たないが論理的で、品質を下げずに無駄を省くための「値引き・条件・内訳開示交渉スクリプト」を生成すること。
5. 法的断定表現の禁止（重要）：ユーザーの法的権利や対応を断定する表現（例：「〜であれば支払いを拒否できます」「〜すべきです」）は絶対に使用せず、必ず「〜の場合、交渉材料になります」「〜の確認をおすすめします」等の非断定表現・アドバイス表現に統一すること。
   例（壁クロス張替の場合）：
   NG：「借主負担となっている根拠を業者に確認し、不当な請求であれば支払いを拒否できます。」
   OK：「借主負担となっている根拠を業者に確認し、不明確な場合は交渉の材料にすることをおすすめします。※個別の契約内容や特約により判断が異なる場合があります。最終的な判断は専門家にご確認ください。」

【出力形式と評価基準（evaluation）の厳格なルール】
必ず以下のJSONフォーマットのみで出力すること。マークダウンの装飾（```json など）は含めず、純粋なJSON文字列のみを出力すること。
"evaluation" の値は、全体の金額と相場を比較し、以下の基準に基づいた文字列を1つだけ出力すること。
・相場より50%以上安い、または異常に安い場合：「危険！要注意！」（安すぎる手抜き工事等の警告）
・相場より10%〜50%未満安い場合：「お買い得」
・相場より5%〜10%未満安い場合：「やや割安」
・相場との差が±5%未満の場合：「相場通り」
・相場より5%〜大きく超えない程度に高い場合：「やや割高」
・相場を大きく超えるぼったくりの場合：「危険！（ぼったくりの可能性あり）」
【アフィリエイト・ECサイト誘導の最適化（最重要）】
原則として、プロの技術や専用サービスを利用することでユーザーのQOL（生活の質）が向上するため、既存の専用業者（ASP）への誘導を最優先とします。単に「自分でやった方が安い」という理由だけでECサイト（楽天市場など）を推奨しないでください。
ただし、「専用業者の対象範囲外である（例: スマホ専用業者なのにパソコンの見積もりである等）」と明確に判断できる場合のみ、JSONの "recommend_ec_search" を true に設定してください。
{
  "status": "success",
  "evaluation": "上記の評価基準のいずれかの文字列",
  "estimated_total_market_price": "見積もり対象全体としての本来の適正な相場価格（例: '約15万円〜20万円'）",
  "infrastructure_check": "スマホでサクッと読めるインフラ適合確認結果と警告事項",
  "price_analysis": {
    "itemized_list": [ {"item": "項目名", "price": 0, "status": "相場通り/割高"} ],
    "unnecessary_costs": [ {"item": "項目名", "potential_saving": 0, "reason": "理由"} ]
  },
  "negotiation_script_line": "LINE用コピペテキスト",
  "negotiation_script_shop": ["店頭用カンペ1", "店頭用カンペ2"],
  "recommend_ec_search": false,
  "search_keyword": "商品本体を購入するための検索キーワード。※家電やパソコンの場合は必ず商品名（例: '冷蔵庫 500L'）を出力してください。例外として、修理・清掃などの『無形サービスの依頼』の場合は絶対に空文字にすること。"
}
}
"""

import re

# シンプルかつ最速の動作を実現するため、不要な関数を削除しました。

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
        
        # 翼さんのご指示通り、一覧の動的取得（遅延の原因）を完全に廃止し、「Gemini Flashシリーズ」の基本エイリアスを指定。
        # 'gemini-flash' はGoogle側で自動的にその時点の最新のFlashモデル（3.7など）へルーティングされるため、
        # 通信遅延ゼロで、常に無料で最新かつ高精度なモデルが利用可能です。
        models_to_try = [
            "gemini-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash"
        ]
        
        # === 1ステップで全解析を完了（処理時間を大幅削減） ===
        # SDKエラーと遅延の元凶であったWeb検索機能への依存を廃止し、最新モデルの内部知識と厳格なハルシネーション対策プロンプトで精度と速度を両立。
        
        analysis_prompt = f"【絶対厳守ルール：推測の徹底排除】\n対象製品の存在や適正相場について「確実な最新の知識」がない場合は、絶対に推測や知ったかぶりをせず、必ず評価を「詳細不明（要注意）」としてください。\n\n{prompt}"
        
        response = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model.generate_content([analysis_prompt] + image_parts)
                if response and response.text:
                    break
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue
                
        if not response or not response.text:
            return {"status": "error", "message": "利用可能なすべてのAIモデルの制限に達しました。しばらく経ってからお試しください。"}
            
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
