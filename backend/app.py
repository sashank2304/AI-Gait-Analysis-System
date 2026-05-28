from flask import Flask, jsonify
from flask_cors import CORS

from inference import GaitInferenceEngine
from routes.analysis import create_analysis_blueprint
from utils.camera import CameraService


def create_app():
    app = Flask(__name__)
    CORS(app)

    inference_engine = GaitInferenceEngine()
    camera_service = CameraService(inference_engine)

    app.config["INFERENCE_ENGINE"] = inference_engine
    app.config["CAMERA_SERVICE"] = camera_service

    app.register_blueprint(
        create_analysis_blueprint(camera_service, inference_engine),
        url_prefix="/api",
    )

    @app.get("/")
    def index():
        return jsonify(
            {
                "name": "AI Gait Analysis Dashboard API",
                "status": "online",
                "docs": {
                    "health": "/api/health",
                    "start_camera": "/api/camera/start",
                    "stop_camera": "/api/camera/stop",
                    "prediction": "/api/prediction",
                    "history": "/api/history",
                    "video_feed": "/api/video_feed",
                },
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
