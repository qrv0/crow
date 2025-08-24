package main

import (
	"encoding/json"
	"fmt"

	"github.com/qrv0/crow/internal/fileformat"
)

func inspectCAWSF(path string) error {
	// Optionally verify checksums per section (from META.checksum_index)

	r, err := fileformat.OpenCAWSF(path)
	if err != nil { return err }
	defer r.Close()
	meta, err := r.SectionUncompressed(fileformat.TypeMeta)
	if err != nil { return err }
	var pretty map[string]any
	if err := json.Unmarshal(meta, &pretty); err == nil {
		b, _ := json.MarshalIndent(pretty, "", "  ")
		fmt.Println("META:")
		fmt.Println(string(b))
		// checksum verification
		if idx, ok := pretty["checksum_index"].(map[string]any); ok {
			fmt.Println("Checksums:")
			for k, v := range idx {
				m, _ := v.(map[string]any)
				fmt.Printf("  section %s: chunks=%v algo=%v\n", k, m["count"], m["algo"])
			}
		}
	} else {
		fmt.Println("META: ", len(meta), "bytes (binary kv)")
	}
	bank, _ := r.Section(fileformat.TypeShardBank)
	fmt.Println("SHARD_BANK:", len(bank), "bytes")
	return nil
}

func inspectGGUF(path string) error {
	info, err := fileformat.InspectGGUF(path)
	if err != nil { return err }
	fmt.Printf("GGUF: magic=%q version=%d\n", string(info.Magic[:]), info.Version)
	return nil
}
