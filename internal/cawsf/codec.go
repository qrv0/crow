package cawsf

import (
	"bytes"
	"fmt"
	"io"

	"github.com/klauspost/compress/zstd"
	lz4 "github.com/pierrec/lz4/v4"
)

// decompressShard decodes shard payload according to comp code: 0=raw, 1=zstd, 2=lz4
func decompressShard(comp uint8, data []byte) ([]byte, error) {
	switch comp {
	case 0:
		return data, nil
	case 1:
		dec, err := zstd.NewReader(nil)
		if err != nil { return nil, err }
		defer dec.Close()
		return dec.DecodeAll(data, nil)
	case 2:
		r := lz4.NewReader(bytes.NewReader(data))
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, r); err != nil { return nil, err }
		return buf.Bytes(), nil
	default:
		return nil, fmt.Errorf("unknown shard comp=%d", comp)
	}
}
