from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"Starting {settings['app_name']}...")
    print(settings["app_description"])


if __name__ == "__main__":
    main()
