package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/qrv0/crow/internal/convert"
	"github.com/qrv0/crow/internal/fileformat"
	"github.com/qrv0/crow/internal/safetensors"

	xxh3 "github.com/zeebo/xxh3"
)

// buildRoutingFromBank builds a ROUTING section from shard bank bytes.
// Layout: dim:uint16, n:uint32, n*u32 shard_ids, n*f32 costs, n*dim*f32 keys
// Keys are deterministic per-shard (L2-normalized),
// costs are proportional to uncompressed shard size in MB.
func buildRoutingFromBank(bank []byte) []byte {
	const dim = 64
	// count shards
	type rec struct{ t uint8; scope uint16; comp uint8; usize uint32; csize uint32; off int }
	var shards []rec
	off := 0
	for off+12 <= len(bank) {
		t := bank[off]
		scope := uint16(bank[off+1]) | uint16(bank[off+2])<<8
		_ = scope
		comp := bank[off+3]
		_ = comp
		usize := binary.LittleEndian.Uint32(bank[off+4:off+8])
		csize := binary.LittleEndian.Uint32(bank[off+8:off+12])
		off += 12
		if off+int(csize) > len(bank) { break }
		shards = append(shards, rec{t: t, scope: scope, comp: comp, usize: usize, csize: csize, off: off})
		off += int(csize)
	}
	n := len(shards)
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, uint16(dim))
	binary.Write(buf, binary.LittleEndian, uint32(n))
	// shard ids [0..n-1]
	for i := 0; i < n; i++ { binary.Write(buf, binary.LittleEndian, uint32(i)) }
	// costs as MB (min clamp)
	for i := 0; i < n; i++ {
		mb := float32(shards[i].usize) / (1024*1024)
		if mb < 0.001 { mb = 0.001 }
		binary.Write(buf, binary.LittleEndian, mb)
	}
	// keys
	for i := 0; i < n; i++ {
		vals := make([]float32, dim)
		var norm float64
		seed := uint32(1469598103) ^ uint32(i*16777619)
		var s uint32 = seed
		for j := 0; j < dim; j++ {
			s = s*1664525 + 1013904223
			f := float32((s%1000)) / 1000.0
			vals[j] = f
			norm += float64(f) * float64(f)
		}
		inv := float32(1.0 / (math.Sqrt(norm) + 1e-8))
		for j := 0; j < dim; j++ { vals[j] *= inv; binary.Write(buf, binary.LittleEndian, vals[j]) }
	}
	return buf.Bytes()
}

func cmdConvert() {
    fs := flag.NewFlagSet("convert", flag.ExitOnError)
    inPath := fs.String("model", "", "path to .safetensors")
    outPath := fs.String("out", "", "output .cawsf")
    rank := fs.Int("rank", 64, "low-rank")
    outlierQ := fs.Float64("outlier-q", 0.999, "outlier quantile")
    pqm := fs.Int("pq-m", 8, "PQ m")
    pqk := fs.Int("pq-k", 256, "PQ k")
    maxLayers := fs.Int("max-layers", 0, "optional: process only first N 2D layers (0=all)")
    maxElems := fs.Int("max-elems", 0, "optional: skip 2D layers with more than N elements (0=no limit)")
    fs.Parse(os.Args[2:])
	if *inPath == "" || *outPath == "" { fmt.Println("usage: crow convert --model x.safetensors --out y.cawsf"); os.Exit(1) }
	st, err := safetensors.Open(*inPath)
	if err != nil { fmt.Fprintf(os.Stderr, "convert: open safetensors: %v\n", err); os.Exit(1) }
    // Build layer specs (2D tensors)
    var shardBlobs [][]byte
    var layers []map[string]any
	meta := map[string]any{
		"format_version": 1,
		"author": "crow",
	}
	// try to capture tokenizer reference and hf config near the model path
	dir := filepath.Dir(*inPath)
	if tok, err := os.ReadFile(filepath.Join(dir, "tokenizer.json")); err == nil && len(tok) > 0 {
		meta["tokenizer"] = "local"
	} else {
		meta["tokenizer"] = "gpt2"
	}
	if cfgBytes, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil {
		var cfg map[string]any
		if json.Unmarshal(cfgBytes, &cfg) == nil {
			meta["hf_config"] = cfg
		}
	}
	if tcfgBytes, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		var tcfg map[string]any
		if json.Unmarshal(tcfgBytes, &tcfg) == nil {
			meta["hf_tokenizer_config"] = tcfg
		}
	}
	scope := uint16(0)
	// deterministic order by name
	names := make([]string, 0, len(st.Tensors))
	for name := range st.Tensors { names = append(names, name) }
	sort.Strings(names)
    processed := 0
    for _, name := range names {
        t := st.Tensors[name]
        if len(t.Meta.Shape) != 2 { continue }
        rows, cols := int(t.Meta.Shape[0]), int(t.Meta.Shape[1])
        nelem := rows*cols
        if *maxElems > 0 && nelem > *maxElems { continue }
        if *maxLayers > 0 && processed >= *maxLayers { break }
        // decode tensor data to float32 considering dtype
        data := bytesToF32WithDtype(t.Data, t.Meta.Dtype, nelem)
        spec := convert.LayerSpec{Name: name, Rows: rows, Cols: cols, Data: data, Scope: scope}
        cfg := convert.Config{Rank: *rank, OutlierQuantile: *outlierQ, PQm: *pqm, PQk: *pqk}
        shs, err := convert.ConvertLayer(spec, cfg)
        if err != nil { fmt.Fprintf(os.Stderr, "convert: layer %s error: %v\n", name, err); os.Exit(1) }
        for _, s := range shs {
            shardBlobs = append(shardBlobs, packShard(s.Type, s.Scope, s.Data))
        }
        layers = append(layers, map[string]any{"scope_id": scope, "name": name, "shape": []int{rows, cols}})
        scope++
        processed++
    }
	meta["layers"] = layers
	// Extract codebooks from R shards and rewrite R payloads to reference shared codebooks
	rewritten, codebooks := rewriteRShardsWithSharedCodebooks(shardBlobs)
	bankBytes := bytesJoin(rewritten)
	// ROUTING: build keys and costs aligned with the final shard bank (cost by shard size)
	routing := buildRoutingFromBank(bankBytes)
	// Build checksum index per section (1 MiB chunks)
	chk := make(map[string]any)
	chk[fmt.Sprint(fileformat.TypeCodebooks)] = rollingXXH3Index(codebooks, 1<<20)
	chk[fmt.Sprint(fileformat.TypeShardBank)] = rollingXXH3Index(bankBytes, 1<<20)
	chk[fmt.Sprint(fileformat.TypeRouting)] = rollingXXH3Index(routing, 1<<20)
	meta["checksum_index"] = chk
	// META as JSON
	metaBytes, _ := json.Marshal(meta)
	// Assemble CAWSF (header + toc + sections)
    writer := fileformat.NewWriter()
    writer.AddSection(fileformat.TypeMeta, metaBytes, 0)
    // Compress CODEBOOKS with zstd (good ratio)
    writer.AddSection(fileformat.TypeCodebooks, codebooks, fileformat.FlagCompZSTD)
    // Compress SHARD_BANK with lz4 (fast decode)
    writer.AddSection(fileformat.TypeShardBank, bankBytes, fileformat.FlagCompLZ4)
    // ROUTING is small; keep raw for simplicity
    writer.AddSection(fileformat.TypeRouting, routing, 0)
	if err := writer.Write(*outPath); err != nil { fmt.Fprintf(os.Stderr, "convert: write %s error: %v\n", *outPath, err); os.Exit(1) }
	fmt.Println("Converted:", *outPath)
}

func bytesToF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func bytesToF32WithDtype(b []byte, dtype string, nelem int) []float32 {
	// supports 'F32', 'F16', 'BF16'
	switch dtype {
	case "F32", "float32":
		return bytesToF32(b)
	case "F16", "float16":
		out := make([]float32, nelem)
		for i := 0; i < nelem; i++ {
			if 2*i+2 > len(b) { break }
			h := uint16(b[2*i]) | uint16(b[2*i+1])<<8
			out[i] = fp16to32(h)
		}
		return out
	case "BF16", "bfloat16":
		out := make([]float32, nelem)
		for i := 0; i < nelem; i++ {
			if 2*i+2 > len(b) { break }
			x := uint32(b[2*i+1])<<8 | uint32(b[2*i])
			// BF16 uses high 16 bits as exponent+high mantissa; map to f32 by shifting
			bits := x << 16
			out[i] = math.Float32frombits(bits)
		}
		return out
	default:
		return bytesToF32(b)
	}
}

func fp16to32(h uint16) float32 {
	s := uint32(h>>15) & 0x1
	e := uint32(h>>10) & 0x1F
	m := uint32(h) & 0x3FF
	var f uint32
	if e == 0 {
		if m == 0 { f = s << 31 } else {
			// subnormal
			e2 := uint32(127 - 15 + 1)
			m2 := m << 13
			for (m2 & (1<<23)) == 0 { m2 <<= 1; e2-- }
			m2 &= (1<<23)-1
			f = (s<<31) | (e2<<23) | m2
		}
	} else if e == 0x1F {
		f = (s<<31) | (0xFF<<23) | (m<<13)
	} else {
		e2 := (e) - 15 + 127
		f = (s<<31) | (e2<<23) | (m<<13)
	}
	return math.Float32frombits(f)
}

func packShard(t uint8, scope uint16, payload []byte) []byte {
	var hdr [12]byte
	hdr[0] = t
	hdr[1] = byte(scope & 0xFF)
	hdr[2] = byte(scope >> 8)
	hdr[3] = 0 // comp raw
	binary.LittleEndian.PutUint32(hdr[4:], uint32(len(payload)))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(len(payload)))
	return append(hdr[:], payload...)
}

// rewriteRShardsWithSharedCodebooks scans R-shards and builds a shared CODEBOOKS section.
// It also rewrites R payloads to drop embedded codebooks, storing only header and codes
// and an extra field codebook_id:uint16 referring to the shared pool.
func rewriteRShardsWithSharedCodebooks(shards [][]byte) (rewritten [][]byte, codebooks []byte) {
	var rInfos []struct{ idx int; scope uint16; rows, cols, d, m, k, n int; cb []byte; codes []byte }
	// We'll collect R info, and keep slots for rewritten blobs by index to preserve order.
	rewritten = make([][]byte, len(shards))
	isR := make([]bool, len(shards))
	for i, blob := range shards {
		if len(blob) < 12 { continue }
		h := blob[:12]
		if h[0] != 1 { // non-R: keep as-is (order preserved)
			rewritten[i] = blob
			continue
		}
		isR[i] = true
		csize := int(binary.LittleEndian.Uint32(h[8:12]))
		payload := blob[12:12+csize]
		if len(payload) < 18 { // rows, cols, d, m, k, n
			rewritten[i] = blob
			continue
		}
		rows := int(binary.LittleEndian.Uint32(payload[0:4]))
		cols := int(binary.LittleEndian.Uint32(payload[4:8]))
		d := int(binary.LittleEndian.Uint16(payload[8:10]))
		m := int(binary.LittleEndian.Uint16(payload[10:12]))
		k := int(binary.LittleEndian.Uint16(payload[12:14]))
		n := int(binary.LittleEndian.Uint32(payload[14:18]))
		dsub := d / m
		cbSz := m * k * dsub * 4
		if 18+cbSz > len(payload) { rewritten[i] = blob; continue }
		cb := payload[18 : 18+cbSz]
		codes := payload[18+cbSz:]
		rInfos = append(rInfos, struct{ idx int; scope uint16; rows, cols, d, m, k, n int; cb []byte; codes []byte }{idx: i, scope: uint16(h[1]) | uint16(h[2])<<8, rows: rows, cols: cols, d: d, m: m, k: k, n: n, cb: cb, codes: codes})
	}
	// dedup codebooks by xxh3 hash
	type entry struct { key uint64; data []byte; id int; d, m, k int }
	var pool []entry
	for _, ri := range rInfos {
		key := xxh3.Hash(ri.cb)
		found := -1
		for j, e := range pool { if e.key == key && bytes.Equal(e.data, ri.cb) { found = j; break } }
		if found < 0 {
			pool = append(pool, entry{key: key, data: append([]byte(nil), ri.cb...), id: len(pool), d: ri.d, m: ri.m, k: ri.k})
			found = len(pool)-1
		}
		// rebuild R payload without codebook, add codebook_id
		pb := new(bytes.Buffer)
		// header rows, cols, d, m, k, n
		binary.Write(pb, binary.LittleEndian, uint32(ri.rows))
		binary.Write(pb, binary.LittleEndian, uint32(ri.cols))
		binary.Write(pb, binary.LittleEndian, uint16(ri.d))
		binary.Write(pb, binary.LittleEndian, uint16(ri.m))
		binary.Write(pb, binary.LittleEndian, uint16(ri.k))
		binary.Write(pb, binary.LittleEndian, uint32(ri.n))
		// add codebook_id
		binary.Write(pb, binary.LittleEndian, uint16(found))
		// append codes only
		pb.Write(ri.codes)
		// build new shard header
		h := shards[ri.idx][:12]
		var nhdr [12]byte
		copy(nhdr[:], h)
		binary.LittleEndian.PutUint32(nhdr[4:], uint32(pb.Len()))
		binary.LittleEndian.PutUint32(nhdr[8:], uint32(pb.Len()))
		rewritten[ri.idx] = append(nhdr[:], pb.Bytes()...)
	}
	// Build CODEBOOKS section as: u16 count; then entries: u16 id; u16 d; u16 m; u16 k; u32 size; bytes
	cbBuf := new(bytes.Buffer)
	binary.Write(cbBuf, binary.LittleEndian, uint16(len(pool)))
	for _, e := range pool {
		binary.Write(cbBuf, binary.LittleEndian, uint16(e.id))
		binary.Write(cbBuf, binary.LittleEndian, uint16(e.d))
		binary.Write(cbBuf, binary.LittleEndian, uint16(e.m))
		binary.Write(cbBuf, binary.LittleEndian, uint16(e.k))
		binary.Write(cbBuf, binary.LittleEndian, uint32(len(e.data)))
		cbBuf.Write(e.data)
	}
	codebooks = cbBuf.Bytes()
	// Fill any remaining nil entries (non-R kept earlier). Concatenate to result slice
	out := make([][]byte, 0, len(rewritten))
	for i := range rewritten { if rewritten[i] != nil { out = append(out, rewritten[i]) } }
	rewritten = out
	return
}

func rollingXXH3Index(data []byte, chunk int) map[string]any {
    if len(data) == 0 {
        return map[string]any{"algo":"xxh3-64","chunk_size":chunk,"count":0,"hashes_hex":[]string{}}
    }
    hashes := make([]string, 0, (len(data)+chunk-1)/chunk)
    for i := 0; i < len(data); i += chunk {
        end := i + chunk
        if end > len(data) { end = len(data) }
        h := xxh3.Hash(data[i:end])
        hashes = append(hashes, fmt.Sprintf("%016x", h))
    }
    return map[string]any{
        "algo": "xxh3-64",
        "chunk_size": chunk,
        "count": len(hashes),
        "hashes_hex": hashes,
    }
}

func bytesJoin(parts [][]byte) []byte {
	var total int
	for _, p := range parts { total += len(p) }
	out := make([]byte, 0, total)
	for _, p := range parts { out = append(out, p...) }
	return out
}
