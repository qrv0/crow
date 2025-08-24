package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/qrv0/crow/internal/downloader"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	switch os.Args[1] {
	case "init":
		cmdInit()
	case "list":
		cmdList()
	case "pull":
		cmdPull()
	case "inspect":
		cmdInspect()
	case "run":
		cmdRun()
	case "convert":
		cmdConvert()
	case "export":
		cmdExport()
	case "export-gguf":
		cmdExportGGUF()
	case "route":
		cmdRoute()
	case "apply":
		cmdApply()
	case "verify":
		cmdVerify()
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Println("crow - CAWSF/GGUF runtime (Go + llama.cpp)")
	fmt.Println("usage: crow <command> [args]")
	fmt.Println("  init                        initialize ~/.crow")
	fmt.Println("  list                        list models in ~/.crow/models")
	fmt.Println("  pull  <url>                 download model file to ~/.crow/models")
    fmt.Println("  inspect <file.{cawsf,gguf}> inspect model file")
    fmt.Println("  run    <file.gguf> [-p prompt] [--ctx 4096] [--gpu-layers N]")
    fmt.Println("  route  --in <file.cawsf> -p 'prompt' [--k 8] [--budget X]")
    fmt.Println("  apply  --in <file.cawsf> --scope N --xlen COLS")
    fmt.Println("  export --in <file.cawsf> --out <dir>            export reconstructed f32 blobs per scope")
    fmt.Println("  export-gguf --in <file.cawsf> --out <file.gguf> export GGUF with f32 tensors")
    fmt.Println("  verify --in <file.cawsf>              verify checksums")
}

var (
	homeDir   = must(os.UserHomeDir())
	crowHome  = filepath.Join(homeDir, ".crow")
	modelsDir = filepath.Join(crowHome, "models")
)

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func cmdInit() {
	if err := os.MkdirAll(modelsDir, 0o755); err != nil { log.Fatal(err) }
	fmt.Println("Initialized:", crowHome)
}

func cmdList() {
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		log.Fatal(err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if filepath.Ext(name) == ".cawsf" || filepath.Ext(name) == ".gguf" {
			fmt.Println(name)
		}
	}
}

func cmdPull() {
	if len(os.Args) < 3 {
		fmt.Println("usage: crow pull <url>")
		os.Exit(1)
	}
	url := os.Args[2]
	out := filepath.Join(modelsDir, filepath.Base(url))
	if err := downloadFile(url, out); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Downloaded:", out)
}

func downloadFile(url, out string) error {
	return downloader.Download(url, out)
}

func cmdInspect() {
	if len(os.Args) < 3 {
		fmt.Println("usage: crow inspect <file.{cawsf,gguf}>")
		os.Exit(1)
	}
	path := os.Args[2]
	switch filepath.Ext(path) {
	case ".cawsf":
		if err := inspectCAWSF(path); err != nil { log.Fatal(err) }
	case ".gguf":
		if err := inspectGGUF(path); err != nil { log.Fatal(err) }
	default:
		fmt.Println("unknown extension")
	}
}

func cmdRun() {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	prompt := fs.String("p", "Hello from crow", "prompt")
	ctxSize := fs.Int("ctx", 4096, "context size")
	gpuLayers := fs.Int("gpu-layers", 0, "GPU layers (llama.cpp)")
	temp := fs.Float64("temperature", 0.8, "sampling temperature")
	topk := fs.Int("top-k", 50, "top-k")
	topp := fs.Float64("top-p", 0.95, "top-p")
	repeat := fs.Float64("repeat-penalty", 1.1, "repeat penalty")
	fs.Parse(os.Args[2:])
	if fs.NArg() < 1 {
		fmt.Println("usage: crow run <file.gguf> [-p prompt] [--ctx 4096] [--gpu-layers N] [--temperature 0.8] [--top-k 50] [--top-p 0.95] [--repeat-penalty 1.1]")
		os.Exit(1)
	}
	modelPath := fs.Arg(0)
	if filepath.Ext(modelPath) != ".gguf" {
		log.Fatal("run currently supports only GGUF; use inspect/convert for CAWSF")
	}
	if err := runGGUFWithSampling(modelPath, *prompt, *ctxSize, *gpuLayers, *temp, *topk, *topp, *repeat); err != nil {
		log.Fatal(err)
	}
}
