package main

import (
    "encoding/json"
    "fmt"
    "path/filepath"
    "testing"

    "github.com/qrv0/crow/internal/fileformat"
    xxh3 "github.com/zeebo/xxh3"
)

// helper: compute rolling xxh3 on data with chunk size
func testRoll(data []byte, chunk int) []uint64 {
    hashes := make([]uint64, 0, (len(data)+chunk-1)/chunk)
    for i := 0; i < len(data); i += chunk {
        end := i + chunk
        if end > len(data) { end = len(data) }
        hashes = append(hashes, xxh3.Hash(data[i:end]))
    }
    return hashes
}

func TestVerifyChecksumsHex(t *testing.T) {
    dir := t.TempDir()
    path := filepath.Join(dir, "toy.cawsf")
    // Build sections
    meta := map[string]any{"format_version": 1}
    code := []byte("codebooks")
    bank := []byte("bank payload with shards")
    rout := []byte("routing payload")
    // Checksums (1 KiB chunk)
    chk := map[string]any{}
    chunk := 1024
    // helper to hex
    toHex := func(v []uint64) []string {
        out := make([]string, len(v))
        for i, x := range v { out[i] = fmt.Sprintf("%016x", x) }
        return out
    }
    chk["2"] = map[string]any{"algo":"xxh3-64","chunk_size":chunk,"count":len(testRoll(code, chunk)),"hashes_hex": toHex(testRoll(code, chunk))}
    chk["3"] = map[string]any{"algo":"xxh3-64","chunk_size":chunk,"count":len(testRoll(bank, chunk)),"hashes_hex": toHex(testRoll(bank, chunk))}
    chk["4"] = map[string]any{"algo":"xxh3-64","chunk_size":chunk,"count":len(testRoll(rout, chunk)),"hashes_hex": toHex(testRoll(rout, chunk))}
    meta["checksum_index"] = chk
    mb, _ := json.Marshal(meta)
    // Write CAWSF
    w := fileformat.NewWriter()
    w.AddSection(fileformat.TypeMeta, mb, 0)
    w.AddSection(fileformat.TypeCodebooks, code, 0)
    w.AddSection(fileformat.TypeShardBank, bank, 0)
    w.AddSection(fileformat.TypeRouting, rout, 0)
    if err := w.Write(path); err != nil { t.Fatalf("write error: %v", err) }
    // Open and verify using same logic
    r, err := fileformat.OpenCAWSF(path)
    if err != nil { t.Fatalf("open error: %v", err) }
    defer r.Close()
    metaBytes, _ := r.SectionUncompressed(fileformat.TypeMeta)
    var m map[string]any
    json.Unmarshal(metaBytes, &m)
    idx := m["checksum_index"].(map[string]any)
    for _, sec := range []uint32{fileformat.TypeCodebooks, fileformat.TypeShardBank, fileformat.TypeRouting} {
        name := fmt.Sprintf("%d", sec)
        mm := idx[name].(map[string]any)
        chunk := int(mm["chunk_size"].(float64))
        want := parseHashes(mm)
        data, _ := r.SectionUncompressed(sec)
        have := rollXXH3(data, chunk)
        if len(have) != len(want) { t.Fatalf("chunk count mismatch") }
        for i := range have {
            if have[i] != want[i] { t.Fatalf("hash mismatch at %d", i) }
        }
    }
}
