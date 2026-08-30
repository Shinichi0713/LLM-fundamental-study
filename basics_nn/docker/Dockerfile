FROM hello-world AS hello-base

FROM python:3.11-alpine
# hello-base ステージから /hello をコピー
COPY --from=hello-base /hello /usr/local/bin/hello
COPY hello.py /app/hello.py
WORKDIR /app
CMD ["sh", "-c", "hello && python hello.py"]