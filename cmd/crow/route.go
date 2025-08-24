package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/qrv0/crow/internal/fileformat"
	xxh3 "github.com/zeebo/xxh3"
)

func cmdRoute() {
	fs := flag.NewFlagSet("route", flag.ExitOnError)
	in := fs.String("in", "", "input .cawsf")
	prompt := fs.String("p", "", "prompt text")
	k := fs.Int("k", 8, "top-k shards to select")
	budget := fs.Float64("budget", 0, "optional budget to respect (0 = ignore)")
	fs.Parse(os.Args[2:])
	if *in == "" || *prompt == "" { fmt.Println("usage: crow route --in model.cawsf -p 'prompt' [--k 8] [--budget 0]"); os.Exit(1) }
	r, err := fileformat.OpenCAWSF(*in)
	if err != nil { fmt.Fprintf(os.Stderr, "route: open error: %v\n", err); os.Exit(1) }
	defer r.Close()
	routing, err := r.SectionUncompressed(fileformat.TypeRouting)
	if err != nil { fmt.Fprintf(os.Stderr, "route: read routing error: %v\n", err); os.Exit(1) }
	dim, n, shardIDs, costs, keys, err := parseRouting(routing)
	if err != nil { fmt.Fprintf(os.Stderr, "route: parse routing error: %v\n", err); os.Exit(1) }
	q := keyFromPrompt(*prompt, dim)
	order := rankByCosine(keys, q)
	selected := make([]int, 0, *k)
	total := 0.0
	for _, idx := range order {
		c := float64(costs[idx])
		if *budget > 0 && total+c > *budget { continue }
		selected = append(selected, int(shardIDs[idx]))
		total += c
		if len(selected) >= *k { break }
	}
	fmt.Printf("Selected %d/%d shards (budget=%.2f)\n", len(selected), n, *budget)
	for i, sid := range selected {
		fmt.Printf("%2d: shard_id=%d cost=%.3f\n", i, sid, costs[order[i]])
	}
}

func parseRouting(data []byte) (dim int, n int, shardIDs []uint32, costs []float32, keys [][]float32, err error) {
	if len(data) < 2+4 { return 0,0,nil,nil,nil, fmt.Errorf("routing: short header") }
	dim = int(binary.LittleEndian.Uint16(data[0:2]))
	n = int(binary.LittleEndian.Uint32(data[2:6]))
	off := 6
	shardIDs = make([]uint32, n)
	for i := 0; i < n; i++ { shardIDs[i] = binary.LittleEndian.Uint32(data[off+4*i:off+4*(i+1)]) }
	off += 4*n
	costs = make([]float32, n)
	for i := 0; i < n; i++ { bits := binary.LittleEndian.Uint32(data[off+4*i:off+4*(i+1)]); costs[i] = math.Float32frombits(bits) }
	off += 4*n
	keys = make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ { v[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:off+4])); off += 4 }
		// keys already normalized in file, but ensure numeric stability
		norm := 0.0
		for j := 0; j < dim; j++ { norm += float64(v[j])*float64(v[j]) }
		inv := float32(1.0/(math.Sqrt(norm)+1e-8))
		for j := 0; j < dim; j++ { v[j] *= inv }
		keys[i] = v
	}
	return
}

func keyFromPrompt(prompt string, dim int) []float32 {
    // Simple hashed bag-of-words projection into dim dims
    // Tokenize by whitespace; for each token, hash and accumulate into one dimension bucket.
    v := make([]float32, dim)
    tokens := strings.Fields(prompt)
    if len(tokens) == 0 { tokens = []string{""} }
    for _, t := range tokens {
        h := xxh3.HashString(t)
        idx := int(h % uint64(dim))
        // pseudo-TF weighting: 1 + log(freq) is approximated by 1 here
        v[idx] += 1.0
    }
    // L2 normalize
    norm := 0.0
    for i := 0; i < dim; i++ { norm += float64(v[i]) * float64(v[i]) }
    inv := float32(1.0 / (math.Sqrt(norm) + 1e-8))
    for i := 0; i < dim; i++ { v[i] *= inv }
    return v
}

func rankByCosine(keys [][]float32, q []float32) []int {
	sims := make([]float64, len(keys))
	for i := range keys {
		s := 0.0
		for j := range q { s += float64(keys[i][j])*float64(q[j]) }
		sims[i] = s
	}
	// argsort descending
	idx := make([]int, len(keys))
	for i := range idx { idx[i] = i }
	// simple selection sort for small n
	for i := 0; i < len(idx); i++ {
		b := i
		for j := i+1; j < len(idx); j++ { if sims[idx[j]] > sims[idx[b]] { b = j } }
		idx[i], idx[b] = idx[b], idx[i]
	}
	return idx
}
