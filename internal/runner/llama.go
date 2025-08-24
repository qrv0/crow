//go:build llama

package runner

import (
	"context"
	"fmt"
	"time"

	llama "github.com/go-skynet/go-llama.cpp"
)

type LLaMARunner struct {
	Model *llama.LLama
}

type RunOptions struct {
	CtxSize   int
	GPULayers int
}

type SampleOptions struct {
	Temperature   float64
	TopK          int
	TopP          float64
	RepeatPenalty float64
}

func New(modelPath string, opt RunOptions) (*LLaMARunner, error) {
	ll, err := llama.New(modelPath,
		llama.SetContext(opt.CtxSize),
		llama.SetGPULayers(opt.GPULayers),
	)
	if err != nil { return nil, err }
	return &LLaMARunner{ Model: ll }, nil
}

func (r *LLaMARunner) Generate(prompt string, s SampleOptions) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	opts := []llama.PredictOption{ llama.Debug(false) }
	if s.Temperature > 0 { opts = append(opts, llama.SetTemperature(float32(s.Temperature))) }
	if s.TopK > 0 { opts = append(opts, llama.SetTopK(s.TopK)) }
	if s.TopP > 0 { opts = append(opts, llama.SetTopP(float32(s.TopP))) }
	if s.RepeatPenalty > 0 { opts = append(opts, llama.SetPenalty(float32(s.RepeatPenalty))) }
	resp, err := r.Model.Predict(ctx, prompt, opts...)
	if err != nil { return "", err }
	return resp, nil
}
