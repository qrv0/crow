package main

import (
	"strings"
	"fmt"
	"github.com/qrv0/crow/internal/runner"
)

func runGGUF(path, prompt string, ctx, gpuLayers int) error {
    // Friendly guard: if llama tag not present, runner.New returns a clear error; make it more explicit
    // We cannot detect build tags at runtime, so we rely on the error string

	r, err := runner.New(path, runner.RunOptions{CtxSize: ctx, GPULayers: gpuLayers})
	if err != nil {
		if strings.Contains(err.Error(), "llama runner unavailable") {
			return fmt.Errorf("run: llama support not built. Rebuild with -tags llama. %w", err)
		}
		return err
	}
	resp, err := r.Generate(prompt, runner.SampleOptions{})
	if err != nil {
		if strings.Contains(err.Error(), "llama runner unavailable") {
			return fmt.Errorf("run: llama support not built. Rebuild with -tags llama. %w", err)
		}
		return err
	}
	fmt.Println(resp)
	return nil
}

func runGGUFWithSampling(path, prompt string, ctx, gpuLayers int, temperature float64, topk int, topp float64, repeatPenalty float64) error {
	r, err := runner.New(path, runner.RunOptions{CtxSize: ctx, GPULayers: gpuLayers})
	if err != nil {
		if strings.Contains(err.Error(), "llama runner unavailable") {
			return fmt.Errorf("run: llama support not built. Rebuild with -tags llama. %w", err)
		}
		return err
	}
	opts := runner.SampleOptions{Temperature: temperature, TopK: topk, TopP: topp, RepeatPenalty: repeatPenalty}
	resp, err := r.Generate(prompt, opts)
	if err != nil {
		if strings.Contains(err.Error(), "llama runner unavailable") {
			return fmt.Errorf("run: llama support not built. Rebuild with -tags llama. %w", err)
		}
		return err
	}
	fmt.Println(resp)
	return nil
}
