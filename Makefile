SHELL := /bin/bash

.PHONY: build build-llama build-cuda test toy docker

build:
	go build -o crow ./cmd/crow

build-llama:
	go build -tags llama -o crow ./cmd/crow

build-cuda:
	go build -tags cuda -o crow ./cmd/crow

test:
	go test ./...

toy:
	go build -o crow ./cmd/crow
	go build -o make_safetensors ./cmd/make_safetensors
	./make_safetensors -out toy.safetensors -rows 8 -cols 16
	./crow convert --model toy.safetensors --out toy.cawsf --rank 2 --outlier-q 0.999 --pq-m 4 --pq-k 8 --max-layers 1
	./crow verify --in toy.cawsf
	./crow export --in toy.cawsf --out out_dir
	./crow export-gguf --in toy.cawsf --out toy.gguf

docker:
	docker build -t crow:latest .
