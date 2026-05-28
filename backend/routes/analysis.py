from flask import Blueprint, Response, jsonify


def create_analysis_blueprint(camera_service, inference_engine):
    bp = Blueprint("analysis", __name__)

    @bp.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "cameraActive": camera_service.is_running,
                "modelReady": True,
                "modelPath": str(inference_engine.model_path),
                "thresholdPath": str(inference_engine.threshold_path),
                "savedThreshold": inference_engine.saved_threshold,
                "sequenceLength": inference_engine.sequence_length,
                "featureCount": inference_engine.feature_count,
            }
        )

    @bp.post("/camera/start")
    def start_camera():
        result = camera_service.start()
        return jsonify(result), 200 if result["ok"] else 500

    @bp.post("/camera/stop")
    def stop_camera():
        camera_service.stop()
        inference_engine.reset()
        return jsonify({"ok": True, "message": "Camera stopped."})

    @bp.post("/reset")
    def reset():
        inference_engine.reset()
        return jsonify({"ok": True, "message": "Inference buffers reset."})

    @bp.get("/prediction")
    def prediction():
        latest = inference_engine.latest_prediction
        latest["cameraActive"] = camera_service.is_running
        return jsonify(latest)

    @bp.get("/history")
    def history():
        return jsonify({"history": list(inference_engine.metric_history)})

    @bp.get("/video_feed")
    def video_feed():
        return Response(
            camera_service.frame_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return bp
