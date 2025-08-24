# Multi-stage build for crow (CPU build by default)
FROM golang:1.22 AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/go/pkg/mod \
    go build -o /out/crow ./cmd/crow

FROM gcr.io/distroless/base-debian12:nonroot
WORKDIR /
COPY --from=build /out/crow /usr/local/bin/crow
USER nonroot
ENTRYPOINT ["/usr/local/bin/crow"]
