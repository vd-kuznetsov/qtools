from functools import lru_cache

from tritonclient.http import InferenceServerClient


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def main():
    triton_client = get_client()
    return triton_client


if __name__ == "__main__":
    main()
