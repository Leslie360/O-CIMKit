import argparse
import base64
import binascii
import json
import mimetypes
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from applications.web_platform.engine import PlatformEngine


class PlatformHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        dashboard = Path(__file__).resolve().parents[1] / "web_dashboard"
        super().__init__(*args, directory=str(dashboard), **kwargs)

    server_version = "O-CIMKit"
    sys_version = ""

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        else:
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data: blob:; connect-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; base-uri 'none'")
        super().end_headers()

    def send_json(self, payload, status=200):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def api_path(self):
        path = unquote(urlsplit(self.path).path)
        host = self.headers.get("Host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost", "[::1]"}:
            raise PermissionError("仅允许本机访问训练引擎")
        return path

    def read_json_body(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("请求必须使用 application/json")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 30 * 1024 * 1024:
            raise ValueError("请求大小无效或超过限制")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("JSON 请求格式错误") from error

    def do_GET(self):
        if not self.path.startswith("/api/"):
            return super().do_GET()
        try:
            path = self.api_path()
            engine = self.server.engine
            if path == "/api/health":
                return self.send_json(engine.health())
            segments = path.strip("/").split("/")
            if len(segments) == 3 and segments[:2] == ["api", "jobs"]:
                return self.send_json(engine.read_job(segments[2]))
            if len(segments) == 3 and segments[:2] == ["api", "results"]:
                return self.send_json(engine.result(segments[2]))
            if len(segments) == 4 and segments[:2] == ["api", "artifacts"]:
                artifact = engine.artifact(segments[2], segments[3])
                content = artifact.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(artifact.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Content-Disposition", f'attachment; filename="O-CIMKit-{segments[2][:8]}-{artifact.name}"')
                self.end_headers()
                return self.wfile.write(content)
            return self.send_json({"error": "接口不存在"}, 404)
        except (ValueError, FileNotFoundError) as error:
            return self.send_json({"error": str(error)}, 400)
        except PermissionError as error:
            return self.send_json({"error": str(error)}, 403)
        except Exception as error:
            return self.send_json({"error": f"服务内部错误：{error}"}, 500)

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self.send_json({"error": "接口不存在"}, 404)
        try:
            path = self.api_path()
            engine = self.server.engine
            if path == "/api/example":
                return self.send_json(engine.prepare_example(), 201)
            payload = self.read_json_body()
            if path == "/api/prepare":
                try:
                    content = base64.b64decode(payload.get("content_base64", ""), validate=True)
                except (binascii.Error, ValueError) as error:
                    raise ValueError("上传文件编码无效") from error
                result = engine.prepare_upload(payload.get("filename", ""), content,
                                               payload.get("measurement_unit", ""))
                return self.send_json(result, 201)
            if path == "/api/jobs":
                result = engine.start_training(
                    payload.get("prepared_id", ""),
                    epochs=payload.get("epochs", 150),
                    seed=payload.get("seed", 42),
                    validation=payload.get("validation", "patient-group-kfold"),
                )
                return self.send_json(result, 202)
            return self.send_json({"error": "接口不存在"}, 404)
        except (ValueError, FileNotFoundError) as error:
            return self.send_json({"error": str(error)}, 400)
        except PermissionError as error:
            return self.send_json({"error": str(error)}, 403)
        except Exception as error:
            return self.send_json({"error": f"服务内部错误：{error}"}, 500)


def main():
    parser = argparse.ArgumentParser(description="O-CIMKit local research platform")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--jobs-root", default=None, help="Override the local job directory for testing")
    parser.add_argument("--open", action="store_true", help="Open the platform in the default browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间")
    project_root = Path(__file__).resolve().parents[2]
    engine = PlatformEngine(project_root, jobs_root=args.jobs_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), PlatformHandler)
    server.engine = engine
    server.daemon_threads = True
    url = f"http://127.0.0.1:{args.port}/"
    print(f"O-CIMKit 本地科研平台：{url}")
    print("真实数据仅在本机处理；公开 GitHub Pages 为只读成果模式。按 Ctrl+C 停止服务。")
    if args.open:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
