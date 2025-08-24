package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/qrv0/crow/internal/fileformat"
	xxh3 "github.com/zeebo/xxh3"
)

func cmdVerify() {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	in := fs.String("in", "", "input .cawsf")
	fs.Parse(os.Args[2:])
	if *in == "" { fmt.Println("usage: crow verify --in model.cawsf"); os.Exit(1) }
	r, err := fileformat.OpenCAWSF(*in)
	if err != nil { fmt.Fprintf(os.Stderr, "verify: open error: %v\n", err); os.Exit(1) }
	defer r.Close()
	metaBytes, _ := r.SectionUncompressed(fileformat.TypeMeta)
	var meta map[string]any
	json.Unmarshal(metaBytes, &meta)
    idx, ok := meta["checksum_index"].(map[string]any)
	if !ok { fmt.Println("no checksum_index in META"); os.Exit(2) }
	okAll := true
	for _, sec := range []uint32{fileformat.TypeCodebooks, fileformat.TypeShardBank, fileformat.TypeRouting} {
		name := fmt.Sprint(sec)
		m, mok := idx[name].(map[string]any)
		if !mok { fmt.Printf("missing checksum for section %s\n", name); okAll = false; continue }
        chunk := int(m["chunk_size"].(float64))
        want := parseHashes(m)
        data, err := r.SectionUncompressed(sec)
		if err != nil { fmt.Printf("read section %s error: %v\n", name, err); okAll = false; continue }
		have := rollXXH3(data, chunk)
		if len(have) != len(want) { fmt.Printf("section %s: chunk count mismatch have %d want %d\n", name, len(have), len(want)); okAll = false; continue }
        for i := range have {
            if have[i] != want[i] { fmt.Printf("section %s: chunk %d mismatch\n", name, i); okAll = false }
        }
	}
	if okAll { fmt.Println("checksum verify: OK") } else { fmt.Fprintln(os.Stderr, "checksum verify: FAILED"); os.Exit(3) }
}

func parseHashes(m map[string]any) []uint64 {
    // prefer hex strings to avoid JSON float precision loss
    if hx, ok := m["hashes_hex"].([]any); ok {
        out := make([]uint64, len(hx))
        for i, v := range hx {
            s := v.(string)
            var x uint64
            fmt.Sscanf(s, "%x", &x)
            out[i] = x
        }
        return out
    }
    // fallback to numeric array (may suffer precision if produced elsewhere)
    if a, ok := m["hashes"].([]any); ok {
        out := make([]uint64, len(a))
        for i, v := range a { out[i] = uint64(v.(float64)) }
        return out
    }
    return nil
}

func rollXXH3(data []byte, chunk int) []uint64 {
	hashes := make([]uint64, 0, (len(data)+chunk-1)/chunk)
	for i := 0; i < len(data); i += chunk {
		end := i + chunk
		if end > len(data) { end = len(data) }
		hashes = append(hashes, xxh3.Hash(data[i:end]))
	}
	return hashes
}
