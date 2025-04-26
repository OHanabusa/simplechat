# lambda/index.py
import json
import os
import re
import urllib.request
import urllib.error


# -----------------------------
#  ユーティリティ
# -----------------------------
def extract_region_from_arn(arn: str) -> str:
    """
    Lambda の ARN からリージョン文字列を抜き出す。
    例:  arn:aws:lambda:us-east-1:123456789012:function:myfn  ->  us-east-1
    """
    m = re.search(r"arn:aws:lambda:([^:]+):", arn)
    return m.group(1) if m else "us-east-1"


# -----------------------------
#  定数・環境変数
# -----------------------------
# CDK / Lambda コンソールで設定した環境変数に
FASTAPI_URL = os.getenv("FASTAPI_URL")
if not FASTAPI_URL:
    raise RuntimeError("環境変数 FASTAPI_URL が設定されていません")



# -----------------------------
#  Lambda ハンドラ
# -----------------------------
def lambda_handler(event, context):
    try:
        print("Call FastAPI endpoint:", FASTAPI_URL)

        # ---------- リクエスト解析 ----------
        body = json.loads(event["body"])
        message = body["message"]
        conversation_history = body.get("conversationHistory", [])

        print("Processing message:", message)

        # ---------- FastAPI へリクエスト ----------
        payload = json.dumps({
            "prompt": message,
            # ↓必要に応じてパラメータを変更
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }).encode("utf-8")

        req = urllib.request.Request(
            FASTAPI_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = json.loads(resp.read().decode("utf-8"))
                # FastAPI 側の response_model に合わせてキーを取得
                assistant_response = (
                    resp_body.get("generated_text")
                    or resp_body.get("text")
                    or resp_body  # 想定外でも丸ごと返す
                )
                print("FastAPI response:", str(assistant_response)[:120])
        except urllib.error.HTTPError as e:
            raise Exception(f"FastAPI returned {e.code}: {e.read().decode()}")
        except urllib.error.URLError as e:
            raise Exception(f"Failed to reach FastAPI: {e.reason}")

        # ---------- 会話履歴を更新 ----------
        conversation_history.append(
            {"role": "user", "content": message}
        )
        conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        # ---------- 正常レスポンス ----------
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers":
                    "Content-Type,X-Amz-Date,Authorization,"
                    "X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": conversation_history
            })
        }

    # ---------------- エラー時 ----------------
    except Exception as err:
        print("Error:", str(err))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers":
                    "Content-Type,X-Amz-Date,Authorization,"
                    "X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({
                "success": False,
                "error": str(err)
            })
        }

