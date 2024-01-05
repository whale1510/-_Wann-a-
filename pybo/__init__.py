from flask import Flask
from flask_cors import CORS
#init파일은 이 형식으로만 작동합니다. __main app형식으로 바꾸지 말것. 오류남.
def create_app():
    app = Flask(__name__)
    CORS(app)  # CORS 적용

    from .views import main_views
    app.register_blueprint(main_views.bp)

    return app