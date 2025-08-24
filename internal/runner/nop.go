package runner

import "fmt"
type LLaMARunner struct{}

type RunOptions struct{ CtxSize, GPULayers int }

type SampleOptions struct{ Temperature float64; TopK int; TopP float64; RepeatPenalty float64 }

func New(modelPath string, opt RunOptions) (*LLaMARunner, error) {
    return nil, fmt.Errorf("llama runner unavailable: build with -tags llama and install go-llama.cpp")
}

func (r *LLaMARunner) Generate(prompt string, s SampleOptions) (string, error) {
    return "", fmt.Errorf("llama runner unavailable: build with -tags llama")
}
