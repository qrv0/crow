package fileformat

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type GGUFInfo struct {
	Magic   [4]byte
	Version uint32
	TensorCount uint64
}

func InspectGGUF(path string) (*GGUFInfo, error) {
	f, err := os.Open(path)
	if err != nil { return nil, err }
	defer f.Close()
	var info GGUFInfo
	if _, err := io.ReadFull(f, info.Magic[:]); err != nil { return nil, err }
	if string(info.Magic[:]) != "GGUF" {
		return nil, fmt.Errorf("not GGUF")
	}
	if err := binary.Read(f, binary.LittleEndian, &info.Version); err != nil { return nil, err }
	// skipping header details; reading tensor count requires parsing metadata fully.
	return &info, nil
}
