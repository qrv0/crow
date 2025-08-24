package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"math"
	"os"
)

// Minimal .safetensors writer with a single F32 tensor
func main() {
	out := flag.String("out", "toy.safetensors", "output safetensors path")
	rows := flag.Int("rows", 8, "rows")
	cols := flag.Int("cols", 16, "cols")
	flag.Parse()
	n := (*rows) * (*cols)
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		data[i] = float32(math.Sin(float64(i))*0.1 + 0.01*float64(i%7))
	}
	// build header
	size := n * 4
	header := map[string]any{
		"toy.weight": map[string]any{
			"dtype":        "F32",
			"shape":        []int{*rows, *cols},
			"data_offsets": []int{0, size},
		},
	}
	hb, _ := json.Marshal(header)
	f, err := os.Create(*out)
	if err != nil {
		println("make_safetensors: create error:", err.Error())
		os.Exit(1)
	}
	defer f.Close()
	bw := bufio.NewWriter(f)
	// write header length (u64 LE)
	var hlen uint64 = uint64(len(hb))
	var b8 [8]byte
	binary.LittleEndian.PutUint64(b8[:], hlen)
	if _, err := bw.Write(b8[:]); err != nil {
		println("make_safetensors: write header length error:", err.Error())
		os.Exit(1)
	}
	if _, err := bw.Write(hb); err != nil {
		println("make_safetensors: write header error:", err.Error())
		os.Exit(1)
	}
	// write tensor data
	for i := 0; i < n; i++ {
		if err := binary.Write(bw, binary.LittleEndian, data[i]); err != nil {
			println("make_safetensors: write data error:", err.Error())
			os.Exit(1)
		}
	}
	if err := bw.Flush(); err != nil {
		println("make_safetensors: flush error:", err.Error())
		os.Exit(1)
	}
}
