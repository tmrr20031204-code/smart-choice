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
1. ハルシネーションの徹底排除：情報が不確定な場合は決して推測せず「判定不能・要確認」と出力すること。
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
  "search_keyword": "楽天市場等で『商品本体を新たに購入する』ための最適な商品キーワード。※注意: 修理、清掃、工事などの『サービスの依頼』に関する見積もりの場合は、商品名が書かれていても絶対に空文字にすること。"
}
"""

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
            category_instruction = "【分析対象: 自動車の車検・整備】法定費用（税金等）と整備費用を分け、過剰な添加剤（エンジン内部洗浄など）、早すぎる消耗品交換（バッテリー、ブレーキパッド、ワイパーゴム等）、不要なコーティング等を見抜いてください。安全性に必要な整備は削らないようにしてください。"
        elif category == "家電（冷蔵庫・洗濯機・掃除機など）":
            category_instruction = "【分析対象: 家電（冷蔵庫・洗濯機・掃除機など）】本体価格や搬入経路（階段昇降費用、クレーン吊り上げ等）、設置費用、リサイクル料金が適正相場か確認してください。\n※【重要: システム連携情報】現在連携している専用業者は「Yahoo!ショッピング（総合EC）」です。家電本体や部品の購入に幅広く対応しているため、原則として 'recommend_ec_search' は false のままにしてください。"
        elif category == "エアコン":
            category_instruction = "【分析対象: エアコン】コンセント形状や電圧（100V/200V）の確認、配管延長費用、高所作業費などの基本・追加工事費が適正か確認してください。"
        elif category == "ハウスクリーニング":
            category_instruction = "【分析対象: ハウスクリーニング】キッチン、換気扇、お風呂、トイレ、床などの清掃費用が適正相場か確認してください。不要なオプション（過剰な防カビコーティングなど）や、不当な出張費・駐車代の請求がないか厳しくチェックしてください。"
        elif category == "賃貸（アパート・マンション等）":
            category_instruction = "【分析対象: 賃貸の初期費用】不動産屋のぼったくりからユーザーを守るため、不要な費用を徹底的に指摘してください。特に「消臭・除菌・抗菌代」「24時間安心サポート」「入居安心サービス」「書類作成代」「法外な鍵交換代」「相場から大きく外れた仲介手数料（原則家賃の0.5ヶ月分）」「指定の割高な火災保険」などは削減・交渉の対象としてください。"
        elif category == "住宅購入・リフォーム":
            category_instruction = "【分析対象: 住宅購入・リフォーム】住宅業者の法外な料金からユーザーを守るため、不明瞭な「諸経費」「書類作成代」「ローン代行手数料」、または相場を大きく超える「オプション工事費」「仲介手数料」を指摘してください。ただし、耐震や構造上の安全性に関わる必須の費用は削らないよう注意してください。"
        elif category == "保険":
            category_instruction = "【分析対象: 保険】不要な特約や、重複している補償内容がないか確認してください。※特別ルール: 画像が「自動車保険」「生命保険」「火災保険」のどれに該当するかを判別し、その結果（例: '自動車保険'）を必ず search_keyword フィールドに出力してください。"
        elif category == "プロパンガス":
            category_instruction = "【分析対象: プロパンガス（LPガス）】基本料金や従量単価（1m3あたり）が地域の適正相場と比較して高すぎないか確認してください。不透明な値上げや不当な請求がないか厳しくチェックしてください。"
        elif category == "電気料金":
            category_instruction = "【分析対象: 電気料金】基本料金や電力量料金（kWh単価）、燃料費調整額などが高すぎないか、または市場連動型プランで高騰していないか確認してください。地域の適正な新電力（固定単価等）と比較して、乗り換えた方がユーザーにとってお得かアドバイスしてください。"
        elif category == "パソコン・スマホ購入・修理":
            category_instruction = "【分析対象: パソコン・スマホ購入・修理】最新機種の新品購入や正規店の高額な修理代の金額が適正か、または高品質な中古品を購入するという別の選択肢と比較してアドバイスしてください。\n※【重要: システム連携情報】現在連携している専用業者は「スマホ・iPhone専用の中古販売業者」です。もしユーザーの画像やテキストが「パソコン（PC）」に関する見積もりである場合は、専用業者の対象外となるため、必ずJSONの 'recommend_ec_search' を true にし、'search_keyword' にパソコンの検索キーワード（例: '中古 ノートパソコン'）を出力してください。"
        else:
            category_instruction = f"【分析対象: {category}】カテゴリに応じた一般的な適正価格と品質維持の観点から分析してください。"

        if text_input:
            prompt = f"以下の見積もりテキスト、および（添付があれば）画像を解析し、指定されたJSON形式で分析結果を出力してください。\n\n【見積もりテキスト】\n{text_input}\n\n{category_instruction}"
        else:
            prompt = f"添付された見積書や設置環境の画像を確認し、以下の指示に従って指定されたJSON形式で分析結果を出力してください。\n\n{category_instruction}"
        
        models_to_try = [
            "gemini-flash-latest",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-3.5-pro", 
            "gemini-3.0-flash"
        ]
        
        # === Step 1: リアルタイムWeb検索による最新相場の取得 ===
        search_prompt = f"以下の見積もり内容および（添付があれば）画像を確認し、Google検索を用いて「今日の最新の適正相場」をリサーチしてください。その上で、見積もりが適正か、ぼったくりか、不要な項目があるかを詳細に分析したテキストレポートを作成してください。\n\n{prompt}"
        
        step1_response_text = ""
        for model_name in models_to_try:
            try:
                search_model = genai.GenerativeModel(
                    model_name=model_name,
                    tools="google_search"
                )
                search_response = search_model.generate_content([search_prompt] + image_parts)
                step1_response_text = search_response.text
                if step1_response_text:
                    break
            except Exception as e:
                continue
                    
        # === フォールバック処理（検索が全滅した場合） ===
        if not step1_response_text:
            response = None
            for model_name in models_to_try:
                try:
                    fallback_model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=SYSTEM_INSTRUCTION,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    response = fallback_model.generate_content([prompt] + image_parts)
                    break
                except Exception:
                    continue
            
            if not response:
                return {"status": "error", "message": "利用可能なすべてのAIモデルの制限に達しました。しばらく経ってからお試しください。"}
            
            try:
                result_json = json.loads(response.text)
            except json.JSONDecodeError:
                return {"status": "error", "message": "AIからの応答を正しく解析できませんでした。"}
        else:
            # === Step 2: 検索結果をJSONフォーマットに整形 ===
            formatting_prompt = f"以下の「分析レポート」をもとに、指定された厳格なJSONフォーマットに整形して出力してください。レポート内の事実に基づき判定してください。\n\n【分析レポート】\n{step1_response_text}"
            
            response = None
            for model_name in models_to_try:
                try:
                    json_model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=SYSTEM_INSTRUCTION,
                        generation_config={"response_mime_type": "application/json"}
                    )
                    response = json_model.generate_content(formatting_prompt)
                    break
                except Exception:
                    continue
                        
            if not response:
                return {"status": "error", "message": "解析中にAIモデルの利用制限に達しました。"}
            
            # 応答テキストをJSONとしてパース
            result_json = json.loads(response.text)
        
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
