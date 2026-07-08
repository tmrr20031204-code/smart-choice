import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
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

【出力形式と評価基準（evaluation）の厳格なルール】
必ず以下のJSONフォーマットのみで出力すること。マークダウンの装飾（```json など）は含めず、純粋なJSON文字列のみを出力すること。
"evaluation" の値は、全体の金額と相場を比較し、以下の基準に基づいた文字列を1つだけ出力すること。
・相場より50%以上安い、または異常に安い場合：「危険！要注意！」（安すぎる手抜き工事等の警告）
・相場より10%〜50%未満安い場合：「お買い得」
・相場より5%〜10%未満安い場合：「やや割安」
・相場との差が±5%未満の場合：「相場通り」
・相場より5%〜大きく超えない程度に高い場合：「やや割高」
・相場を大きく超えるぼったくりの場合：「危険（ぼったくり）」
{
  "status": "success",
  "evaluation": "上記の評価基準のいずれかの文字列",
  "infrastructure_check": "スマホでサクッと読めるインフラ適合確認結果と警告事項",
  "price_analysis": {
    "itemized_list": [ {"item": "項目名", "price": 0, "status": "相場通り/割高"} ],
    "unnecessary_costs": [ {"item": "項目名", "potential_saving": 0, "reason": "理由"} ]
  },
  "negotiation_script_line": "LINE用コピペテキスト",
  "negotiation_script_shop": ["店頭用カンペ1", "店頭用カンペ2"]
}
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

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
        elif category == "大型家電（冷蔵庫・洗濯機等）":
            category_instruction = "【分析対象: 大型家電】搬入経路（階段昇降費用、クレーン吊り上げ等）、設置費用、リサイクル料金が適正相場か確認してください。"
        elif category == "エアコン":
            category_instruction = "【分析対象: エアコン】コンセント形状や電圧（100V/200V）の確認、配管延長費用、高所作業費などの基本・追加工事費が適正か確認してください。"
        elif category == "賃貸（アパート・マンション等）":
            category_instruction = "【分析対象: 賃貸の初期費用】不動産屋のぼったくりからユーザーを守るため、不要な費用を徹底的に指摘してください。特に「消臭・除菌・抗菌代」「24時間安心サポート」「入居安心サービス」「書類作成代」「法外な鍵交換代」「相場から大きく外れた仲介手数料（原則家賃の0.5ヶ月分）」「指定の割高な火災保険」などは削減・交渉の対象としてください。"
        elif category == "住宅購入・リフォーム":
            category_instruction = "【分析対象: 住宅購入・リフォーム】住宅業者の法外な料金からユーザーを守るため、不明瞭な「諸経費」「書類作成代」「ローン代行手数料」、または相場を大きく超える「オプション工事費」「仲介手数料」を指摘してください。ただし、耐震や構造上の安全性に関わる必須の費用は削らないよう注意してください。"
        else:
            category_instruction = f"【分析対象: {category}】カテゴリに応じた一般的な適正価格と品質維持の観点から分析してください。"

        if text_input:
            prompt = f"以下の見積もりテキスト、および（添付があれば）画像を解析し、指定されたJSON形式で分析結果を出力してください。\n\n【見積もりテキスト】\n{text_input}\n\n{category_instruction}"
        else:
            prompt = f"添付された見積書や設置環境の画像を確認し、以下の指示に従って指定されたJSON形式で分析結果を出力してください。\n\n{category_instruction}"
        
        models_to_try = [
            "gemini-3.5-flash", 
            "gemini-3.5-pro", 
            "gemini-3.0-flash", 
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ]
        
        response = None
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model.generate_content([prompt] + image_parts)
                break  # 成功したらループを抜ける
            except Exception as e:
                error_msg = str(e)
                # 429(制限) や 404(存在しないモデル) などの場合は次のモデルを試す
                if "429" in error_msg or "Quota exceeded" in error_msg or "404" in error_msg or "not found" in error_msg.lower() or "deprecated" in error_msg.lower():
                    continue
                else:
                    # それ以外のエラー（認証エラーなど）はそのまま返す
                    return {
                        "status": "error",
                        "message": f"モデル {model_name} でエラーが発生しました: {error_msg}"
                    }
                    
        if not response:
            return {
                "status": "error",
                "message": "利用可能なすべての無料AIモデル（3.5〜2.0シリーズ等）で利用制限（1日の上限）に達したか、利用できません。明日までお待ちいただくか、新しいAPIキーをご用意ください。"
            }
        
        # 応答テキストをJSONとしてパース
        result_json = json.loads(response.text)
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
