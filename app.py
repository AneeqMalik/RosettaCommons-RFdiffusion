import os

from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from src.inference import load_model, predict


app = Flask(__name__)

# Load the model by reading the `SM_MODEL_DIR` environment variable
# which is passed to the container by SageMaker (usually /opt/ml/model).
model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
model = load_model(model_dir)

# Since the web application runs behind a proxy (nginx), we need to
# add this setting to our app.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Function which responds to the invocations requests.
    """
    try:
        body = request.get_json(force=True, silent=False)
    except Exception as exc:
        return jsonify({"error": "Invalid JSON payload", "details": str(exc)}), 400

    try:
        result = predict(body, model)
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        details = exc.args[1] if len(exc.args) > 1 else {"message": str(exc)}
        return (
            jsonify(
                {
                    "error": "RFdiffusion execution failed",
                    "details": details,
                }
            ),
            500,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return (
            jsonify({"error": "Unhandled server error", "details": str(exc)}),
            500,
        )

    return jsonify(result)