package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

	"github.com/qrv0/crow/internal/cawsf"
	"github.com/qrv0/crow/internal/fileformat"
)

// Exporta CAWSF -> GGUF gerando um arquivo GGUF completo com metadados e tensores f32.
// Usa um writer GGUF minimalista compatível com leitores básicos.

func cmdExportGGUF() {
	fs := flag.NewFlagSet("export-gguf", flag.ExitOnError)
	inPath := fs.String("in", "", "input .cawsf")
	outPath := fs.String("out", "", "output .gguf")
	family := fs.String("family", "crow-generic", "model family tag")
	fs.Parse(os.Args[2:])
	if *inPath == "" || *outPath == "" {
		fmt.Println("usage: crow export-gguf --in model.cawsf --out model.gguf [--family crow-generic]")
		os.Exit(1)
	}
	r, err := fileformat.OpenCAWSF(*inPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "export-gguf: open error: %v\n", err)
		os.Exit(1)
	}
	defer r.Close()
	metaBytes, _ := r.SectionUncompressed(fileformat.TypeMeta)
	var meta map[string]any
	json.Unmarshal(metaBytes, &meta)
	bank, err := r.SectionUncompressed(fileformat.TypeShardBank)
	if err != nil {
		fmt.Fprintf(os.Stderr, "export-gguf: read shard bank error: %v\n", err)
		os.Exit(1)
	}
	codebooks, _ := r.SectionUncompressed(fileformat.TypeCodebooks)
	pool, _ := cawsf.ParseCodebookPool(codebooks)
	idx, _ := cawsf.IndexShardBank(bank)
	scopes := map[uint16]struct{}{}
	for _, rec := range idx.Records {
		scopes[rec.Hdr.Scope] = struct{}{}
	}
	// Build GGUF
	gw := fileformat.NewGGUFWriter()
	// Minimal metadata
	modelName := "crow"
	if n, ok := meta["model_name"].(string); ok {
		modelName = n
	}
	gw.AddKV(fileformat.GGUFKV{Key: "general.name", Type: fileformat.GGUFTypeString, Value: modelName})
	gw.AddKV(fileformat.GGUFKV{Key: "general.file_type", Type: fileformat.GGUFTypeUint32, Value: uint32(0)})
	// architecture mapping
	arch := *family
	if hf, ok := meta["hf_config"].(map[string]any); ok {
		if mt, ok := hf["model_type"].(string); ok {
			switch mt {
			case "llama":
				arch = "llama"
			case "mistral":
				arch = "mistral"
			case "qwen2":
				arch = "qwen2"
			case "mixtral":
				arch = "mixtral"
			}
		}
	}
	gw.AddKV(fileformat.GGUFKV{Key: "general.architecture", Type: fileformat.GGUFTypeString, Value: arch})
	// tokenizer metadata (basic)
	if tcfg, ok := meta["hf_tokenizer_config"].(map[string]any); ok {
		if bos, ok := tcfg["bos_token"].(map[string]any); ok {
			if s, ok := bos["content"].(string); ok {
				gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.bos_token", Type: ggufTypeString(), Value: s})
			}
		}
		if eos, ok := tcfg["eos_token"].(map[string]any); ok {
			if s, ok := eos["content"].(string); ok {
				gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.eos_token", Type: ggufTypeString(), Value: s})
			}
		}
		if unk, ok := tcfg["unk_token"].(map[string]any); ok {
			if s, ok := unk["content"].(string); ok {
				gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.unknown_token", Type: ggufTypeString(), Value: s})
			}
		}
		if pad, ok := tcfg["pad_token"].(map[string]any); ok {
			if s, ok := pad["content"].(string); ok {
				gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.pad_token", Type: ggufTypeString(), Value: s})
			}
		}
	}
	if hf, ok := meta["hf_config"].(map[string]any); ok {
		if id, ok := num(hf["bos_token_id"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.ggml.bos_token_id", Type: fileformat.GGUFTypeUint32, Value: uint32(id)})
		}
		if id, ok := num(hf["eos_token_id"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.ggml.eos_token_id", Type: fileformat.GGUFTypeUint32, Value: uint32(id)})
		}
	}
	// Add architecture-specific GGUF metadata if available
	if hf, ok := meta["hf_config"].(map[string]any); ok {
		addGGUFArchMeta(arch, hf, gw)
	}
	// Serialize tensors ordered by scope id for stability with canonical names when possible
	type pair struct {
		sc   uint16
		name string
	}
	var order []pair
	for sc := range scopes {
		order = append(order, pair{sc: sc, name: lookupTensorName(meta, int(sc))})
	}
	sort.Slice(order, func(i, j int) bool { return order[i].sc < order[j].sc })
	for _, p := range order {
		rows, cols, data, err := cawsf.ReconstructForScopeWithPool(bank, pool, p.sc)
		if err != nil {
			fmt.Println("scope", p.sc, "error:", err)
			continue
		}
		name := p.name
		if name == "" {
			name = fmt.Sprintf("scope_%d", p.sc)
		}
		name = canonicalTensorName(arch, name)
		if arch == "qwen2" {
			// Qwen2 usa weight col-major em alguns leitores; se necessário, transpor
			// Por segurança, preservamos row-major aqui; leitores devem lidar com ggml dims/ordem.
		}
		gw.AddTensor(fileformat.GGUFTensor{
			Name: name,
			Dims: []uint64{uint64(rows), uint64(cols)},
			Type: 0, // f32
			Data: f32ToBytes(data),
		})
	}
	// Write GGUF to file
	f, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "export-gguf: create %s error: %v\n", *outPath, err)
		os.Exit(1)
	}
	defer f.Close()
	if err := gw.Write(f); err != nil {
		fmt.Fprintf(os.Stderr, "export-gguf: write error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Exported:", *outPath)
}

// keep a simple sanitization helper if names need normalization in future
func sanitizeName(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "/", "_"), ".", "_")
}

func lookupTensorName(meta map[string]any, scope int) string {
	layers, ok := meta["layers"].([]any)
	if !ok {
		return fmt.Sprintf("scope_%d", scope)
	}
	for _, v := range layers {
		m, _ := v.(map[string]any)
		sid, _ := m["scope_id"].(float64)
		if int(sid) == scope {
			if n, ok := m["name"].(string); ok {
				return n
			}
		}
	}
	return fmt.Sprintf("scope_%d", scope)
}

// expose gguf type id for string without importing constants here
func ggufTypeString() uint32  { return 8 }
func ggufTypeUint32() uint32  { return 4 }
func ggufTypeInt32() uint32   { return 5 }
func ggufTypeFloat32() uint32 { return 6 }
func ggufTypeBool() uint32    { return 7 }

// addGGUFArchMeta injects common KV pairs used by readers for known architectures
func addGGUFArchMeta(arch string, hf map[string]any, gw *fileformat.GGUFWriter) {
	// We add minimal safe keys; readers may still require more specifics.
	switch arch {
	case "llama", "mistral", "mixtral", "qwen2":
		if v, ok := num(hf["vocab_size"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: "tokenizer.ggml.tokens", Type: ggufTypeUint32(), Value: uint32(v)})
		}
		if d, ok := num(hf["hidden_size"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".embedding_length", Type: fileformat.GGUFTypeUint32, Value: uint32(d)})
		}
		if n, ok := num(hf["num_hidden_layers"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".block_count", Type: fileformat.GGUFTypeUint32, Value: uint32(n)})
		}
		if h, ok := num(hf["num_attention_heads"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".attention.head_count", Type: ggufTypeUint32(), Value: uint32(h)})
		}
		if kv, ok := num(hf["num_key_value_heads"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".attention.head_count_kv", Type: ggufTypeUint32(), Value: uint32(kv)})
		}
		if ctx, ok := num(hf["max_position_embeddings"]); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".context_length", Type: ggufTypeUint32(), Value: uint32(ctx)})
		}
		if eps, ok := hf["rms_norm_eps"].(float64); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".attention.layer_norm_rms_eps", Type: ggufTypeFloat32(), Value: float32(eps)})
		}
		if th, ok := hf["rope_theta"].(float64); ok {
			gw.AddKV(fileformat.GGUFKV{Key: arch + ".rope.freq_base", Type: ggufTypeFloat32(), Value: float32(th)})
		}
	}
}

func num(v any) (int, bool) {
	switch t := v.(type) {
	case float64:
		return int(t), true
	case int:
		return t, true
	default:
		return 0, false
	}
}

// canonicalTensorName maps common HF names to GGUF-expected tensor names per arch when possible
func canonicalTensorName(arch, name string) string {
	s := name
	switch arch {
	case "llama", "mistral":
		s = strings.ReplaceAll(s, "model.layers.", "blk.")
		s = strings.ReplaceAll(s, "input_layernorm.weight", "attn_norm.weight")
		s = strings.ReplaceAll(s, "post_attention_layernorm.weight", "ffn_norm.weight")
		s = strings.ReplaceAll(s, "self_attn.q_proj.weight", "attn_q.weight")
		s = strings.ReplaceAll(s, "self_attn.k_proj.weight", "attn_k.weight")
		s = strings.ReplaceAll(s, "self_attn.v_proj.weight", "attn_v.weight")
		s = strings.ReplaceAll(s, "self_attn.o_proj.weight", "attn_output.weight")
		s = strings.ReplaceAll(s, "mlp.gate_proj.weight", "ffn_gate.weight")
		s = strings.ReplaceAll(s, "mlp.up_proj.weight", "ffn_up.weight")
		s = strings.ReplaceAll(s, "mlp.down_proj.weight", "ffn_down.weight")
		s = strings.ReplaceAll(s, "model.embed_tokens.weight", "token_embd.weight")
		s = strings.ReplaceAll(s, "lm_head.weight", "output.weight")
	case "qwen2":
		s = qwen2MapName(s)
	case "mixtral":
		// Mixtral MoE canonicalization
		s = strings.ReplaceAll(s, "model.layers.", "blk.")
		s = strings.ReplaceAll(s, "input_layernorm.weight", "attn_norm.weight")
		s = strings.ReplaceAll(s, "post_attention_layernorm.weight", "ffn_norm.weight")
		// router/gate
		s = strings.ReplaceAll(s, "mlp.gate.weight", "ffn_router.weight")
		s = strings.ReplaceAll(s, "expert_gate.weight", "attn_gate.weight")
		// experts
		s = strings.ReplaceAll(s, ".mlp.experts.", ".ffn_experts.")
		s = strings.ReplaceAll(s, ".w1.weight", ".up.weight")
		s = strings.ReplaceAll(s, ".w3.weight", ".gate.weight")
		s = strings.ReplaceAll(s, ".w2.weight", ".down.weight")
		// Attention projections
		s = strings.ReplaceAll(s, "self_attn.q_proj.weight", "attn_q.weight")
		s = strings.ReplaceAll(s, "self_attn.k_proj.weight", "attn_k.weight")
		s = strings.ReplaceAll(s, "self_attn.v_proj.weight", "attn_v.weight")
		s = strings.ReplaceAll(s, "self_attn.o_proj.weight", "attn_output.weight")
	}
	return s
}

// qwen2MapName converts common HF tensor names for Qwen2 to GGUF‑style canonical names
func qwen2MapName(s string) string {
	// Normalize separators for easier replacement
	s = strings.ReplaceAll(s, ".", "/")
	// Qwen2 typical patterns to canonical equivalents
	replacers := [][2]string{
		{"model/embeddings/word_embeddings/weight", "token_embd.weight"},
		{"lm_head/weight", "output.weight"},
		// RMSNorms
		{"input_layernorm/weight", "attn_norm.weight"},
		{"post_attention_layernorm/weight", "ffn_norm.weight"},
		// Attention projections
		{"self_attn/q_proj/weight", "attn_q.weight"},
		{"self_attn/k_proj/weight", "attn_k.weight"},
		{"self_attn/v_proj/weight", "attn_v.weight"},
		{"self_attn/o_proj/weight", "attn_output.weight"},
		// MLP projections
		{"mlp/gate_proj/weight", "ffn_gate.weight"},
		{"mlp/up_proj/weight", "ffn_up.weight"},
		{"mlp/down_proj/weight", "ffn_down.weight"},
	}
	for _, p := range replacers {
		s = strings.ReplaceAll(s, p[0], p[1])
	}
	// layer index: model/layers/N/... -> blk.N/...
	re := regexp.MustCompile(`model/layers/(\d+)/`)
	s = re.ReplaceAllString(s, "blk.$1/")
	// Back to dot separators
	s = strings.ReplaceAll(s, "/", ".")
	return s
}
