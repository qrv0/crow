package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/qrv0/crow/internal/cawsf"
	"github.com/qrv0/crow/internal/fileformat"
)

func cmdApply() {
	fs := flag.NewFlagSet("apply", flag.ExitOnError)
	in := fs.String("in", "", "input .cawsf")
	scope := fs.Int("scope", -1, "scope id to apply")
	xlen := fs.Int("xlen", 0, "length of input vector (must match cols)")
	fs.Parse(os.Args[2:])
	if *in == "" || *scope < 0 || *xlen <= 0 {
		fmt.Println("usage: crow apply --in model.cawsf --scope N --xlen COLS")
		os.Exit(1)
	}
	r, err := fileformat.OpenCAWSF(*in)
	if err != nil { fmt.Fprintf(os.Stderr, "apply: open error: %v\n", err); os.Exit(1) }
	defer r.Close()
	bank, _ := r.SectionUncompressed(fileformat.TypeShardBank)
	codebooks, _ := r.SectionUncompressed(fileformat.TypeCodebooks)
	pool, _ := cawsf.ParseCodebookPool(codebooks)
	// build a simple deterministic x vector of length xlen
	x := make([]float32, *xlen)
	for i := range x {
		// quasi-random
		u := uint32(2166136261 + i*16777619)
		x[i] = math.Float32frombits(u)
	}
	y, rows, cols, err := cawsf.MultiplyScopeWithPool(bank, pool, uint16(*scope), x)
	if err != nil { fmt.Fprintf(os.Stderr, "apply: compute error: %v\n", err); os.Exit(1) }
	if cols != *xlen { fmt.Printf("warning: xlen=%d but cols=%d\n", *xlen, cols) }
	fmt.Printf("y (rows=%d):\n", rows)
	// print first 16
	n := 16
	if rows < n { n = rows }
	for i := 0; i < n; i++ { fmt.Printf("  y[%d]=%.6f\n", i, y[i]) }
}
