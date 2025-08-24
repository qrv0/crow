package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/qrv0/crow/internal/cawsf"
	"github.com/qrv0/crow/internal/fileformat"
)

// Export CAWSF -> dense .f32 blobs per-layer (row-major).
func cmdExport() {
	fs := flag.NewFlagSet("export", flag.ExitOnError)
	inPath := fs.String("in", "", "input .cawsf")
	outDir := fs.String("out", "", "output dir (.f32 blobs)")
	scope := fs.Int("scope", -1, "export only this scope (optional)")
	fs.Parse(os.Args[2:])
	if *inPath == "" || *outDir == "" { fmt.Println("usage: crow export --in file.cawsf --out dir [--scope N]"); os.Exit(1) }
	r, err := fileformat.OpenCAWSF(*inPath)
	if err != nil { fmt.Fprintf(os.Stderr, "export: open error: %v\n", err); os.Exit(1) }
	defer r.Close()
	bank, err := r.SectionUncompressed(fileformat.TypeShardBank)
	if err != nil { fmt.Fprintf(os.Stderr, "export: read shard bank error: %v\n", err); os.Exit(1) }
    codebooks, _ := r.SectionUncompressed(fileformat.TypeCodebooks)
	pool, _ := cawsf.ParseCodebookPool(codebooks)
	if err := os.MkdirAll(*outDir, 0o755); err != nil { fmt.Fprintf(os.Stderr, "export: mkdir error: %v\n", err); os.Exit(1) }
	idx, _ := cawsf.IndexShardBank(bank)
	scopes := map[uint16]struct{}{}
	for _, rec := range idx.Records { scopes[rec.Hdr.Scope] = struct{}{} }
	for sc := range scopes {
		if *scope >= 0 && int(sc) != *scope { continue }
		rows, cols, data, err := cawsf.ReconstructForScopeWithPool(bank, pool, sc)
		if err != nil { fmt.Println("scope", sc, "error:", err); continue }
		out := filepath.Join(*outDir, fmt.Sprintf("scope_%d_%dx%d.f32", sc, rows, cols))
		if err := os.WriteFile(out, f32ToBytes(data), 0o644); err != nil { fmt.Fprintf(os.Stderr, "export: write %s error: %v\n", out, err); os.Exit(1) }
		fmt.Println("wrote", out)
	}
}

func f32ToBytes(a []float32) []byte {
	b := make([]byte, 4*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}
