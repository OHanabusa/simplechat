# lambda/index.py
import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError

# ------------------------------------------------------------
# ❶ 定数・グローバル
# ------------------------------------------------------------
# Colab + ngrok で公開している FastAPI の /generate エンドポイント
FASTAPI_URL = os.getenv("FASTAPI_URL", "").rstrip("/")  # "" なら FastAPI なし

# フォールバックで使う Bedrock モデル（環境変数で上書き可）
MODEL_ID = os.getenv("MODEL_ID", "us.amazon.nova-lite-v1:0")

# Bedrock runtime client は必要になるまで生成しない
bedrock_client = None

# ------------------------------------------------------------
# ❷ 補助関数
# ------------------------------------------------------------
def extract_region_from_arn(arn: str) -> str:
    """
    Lambda の ARN からリージョンを取り出す。
    arn:aws:lambda:{region}:{account-id}:function:{function-name}
    """
    match = re.search(r"arn:aws:lambda:([^:]+):", arn)
    return match.group(1) if match else "us-east-1"


# ------------------------------------------------------------
# ❸ エントリポイント
# ------------------------------------------------------------
def lambda_handler(event: Dict[str, Any], context):
    global bedrock_client

    print("Received event:", json.dumps(event)[:500])

    # -------------- ユーザー情報（Cognito 経由の場合のみ） --------------
    user_info = None
    if event.get("requestContext", {}).get("authorizer"):
        user_info = event["requestContext"]["authorizer"]["claims"]
        print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")

    # -------------- リクエストパラメータ --------------
    body = json.loads(event["body"])
    message: str = body["message"]
    conversation_history: List[Dict[str, str]] = body.get("conversationHistory", [])

    print("Incoming message:", message[:120])
    print(f"FastAPI URL: {FASTAPI_URL or '(not set)'}")

    # -------------- まず FastAPI へ --------------
    assistant_response = ""
    used_fallback = False

    if FASTAPI_URL:
        try:
            payload = json.dumps(
                {
                    "prompt": message,
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                FASTAPI_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                assistant_response = resp_json.get("generated_text") or resp_json.get("text", "")
                print("FastAPI OK")

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"FastAPI NG → fallback, reason: {e}")
            used_fallback = True
    else:
        used_fallback = True

    # -------------- FastAPI が使えなければ Bedrock へ --------------
    if used_fallback:
        if bedrock_client is None:
            region = extract_region_from_arn(context.invoked_function_arn)
            bedrock_client = boto3.client("bedrock-runtime", region_name=region)
            print(f"Initialised Bedrock client in region: {region}")

        bedrock_messages = [
            {
                "role": "user",
                "content": [{"text": message}],
            }
        ]

        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {"maxTokens": 512, "temperature": 0.7, "topP": 0.9},
        }

        print(f"Calling Bedrock model: {MODEL_ID}")
        bed_resp = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json",
        )
        bed_json = json.loads(bed_resp["body"].read())
        assistant_response = bed_json["output"]["message"]["content"][0]["text"]

    # -------------- 会話履歴に追加 --------------
    conversation_history.append({"role": "assistant", "content": assistant_response})

    # -------------- レスポンス --------------
    print(f"→ Returned by {'Bedrock' if used_fallback else 'FastAPI'}")
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods": "OPTIONS,POST",
        },
        "body": json.dumps(
            {
                "success": True,
                "response": assistant_response,
                "conversationHistory": conversation_history,
                "via": "bedrock" if used_fallback else "fastapi",
            }
        ),
    }
