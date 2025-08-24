package fileformat

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/klauspost/compress/zstd"
	lz4 "github.com/pierrec/lz4/v4"
)

type Writer struct {
	sections []struct{ TypeID uint32; Data []byte; Flags uint32 }
}

func NewWriter() *Writer { return &Writer{} }

func (w *Writer) AddSection(t uint32, data []byte, flags uint32) { w.sections = append(w.sections, struct{TypeID uint32; Data []byte; Flags uint32}{t, data, flags}) }

// Optional compressors for sections
func zstdEncode(b []byte) ([]byte, error) {
	enc, err := zstd.NewWriter(nil)
	if err != nil { return nil, err }
	defer enc.Close()
	return enc.EncodeAll(b, make([]byte, 0, len(b))), nil
}

func zstdDecode(b []byte) ([]byte, error) {
	dec, err := zstd.NewReader(nil)
	if err != nil { return nil, err }
	defer dec.Close()
	out, err := dec.DecodeAll(b, nil)
	return out, err
}

func lz4Encode(b []byte) ([]byte, error) {
	var buf bytes.Buffer
	w := lz4.NewWriter(&buf)
	if _, err := w.Write(b); err != nil { return nil, err }
	if err := w.Close(); err != nil { return nil, err }
	return buf.Bytes(), nil
}

func lz4Decode(b []byte) ([]byte, error) {
	r := lz4.NewReader(bytes.NewReader(b))
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil { return nil, err }
	return buf.Bytes(), nil
}

func alignUp(x, a int64) int64 { r := x % a; if r == 0 { return x }; return x + (a - r) }

func (w *Writer) Write(path string) error {
	f, err := os.Create(path)
	if err != nil { return err }
	defer f.Close()
	// prepare payloads with optional compression according to flags
	payloads := make([][]byte, len(w.sections))
	for i, s := range w.sections {
		data := s.Data
		if s.Flags&FlagCompZSTD != 0 {
			if enc, err := zstdEncode(data); err == nil { data = enc } else { return err }
		} else if s.Flags&FlagCompLZ4 != 0 {
			if enc, err := lz4Encode(data); err == nil { data = enc } else { return err }
		}
		payloads[i] = data
	}
	// header
	if _, err := f.Write(magic[:]); err != nil { return err }
	var hdr struct { Ver, Num, Res uint32 }
	hdr.Ver, hdr.Num, hdr.Res = 1, uint32(len(w.sections)), 0
	if err := binary.Write(f, binary.LittleEndian, &hdr); err != nil { return err }
	// toc
	type rec struct { TypeID uint32; Offset uint64; Size uint64; Flags uint32 }
	recs := make([]rec, len(w.sections))
	// align sections to 4096 bytes after header+toc
	base := int64(8 + 12 + 24*len(w.sections))
	offset := alignUp(base, 4096)
	for i, s := range w.sections {
		data := payloads[i]
		recs[i] = rec{TypeID: s.TypeID, Offset: uint64(offset), Size: uint64(len(data)), Flags: s.Flags}
		offset = alignUp(offset+int64(len(data)), 4096)
	}
	for _, r := range recs {
		if err := binary.Write(f, binary.LittleEndian, &r); err != nil { return err }
	}
	// pad to first section offset
	cur, _ := f.Seek(0, io.SeekCurrent)
	first := int64(recs[0].Offset)
	if cur < first {
		pad := make([]byte, first-cur)
		if _, err := f.Write(pad); err != nil { return err }
	}
	// sections
	for i := range w.sections {
		if _, err := f.Seek(int64(recs[i].Offset), io.SeekStart); err != nil { return err }
		if _, err := f.Write(payloads[i]); err != nil { return err }
	}
	return nil
}

var (
	magic = [8]byte{'C','A','W','S','F',0,0,0}
)

const (
	TypeMeta       = 1
	TypeCodebooks  = 2
	TypeShardBank  = 3
	TypeRouting    = 4
)

type tocEntry struct {
	TypeID uint32
	Offset uint64
	Size   uint64
	Flags  uint32
}

type Reader struct {
	f    *os.File
	TOC  []tocEntry
}

const (
	FlagCompZSTD uint32 = 1 << 0
	FlagCompLZ4  uint32 = 1 << 1
)

func OpenCAWSF(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil { return nil, err }
	head := make([]byte, 8)
	if _, err := io.ReadFull(f, head); err != nil { f.Close(); return nil, err }
	if !bytes.Equal(head, magic[:]) { f.Close(); return nil, errors.New("not a CAWSF file") }
	var hdr struct { Ver, Num, Res uint32 }
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil { f.Close(); return nil, err }
	TOC := make([]tocEntry, hdr.Num)
	for i := 0; i < int(hdr.Num); i++ {
		if err := binary.Read(f, binary.LittleEndian, &TOC[i]); err != nil { f.Close(); return nil, err }
	}
	return &Reader{ f: f, TOC: TOC }, nil
}

func (r *Reader) Close() error { return r.f.Close() }

func (r *Reader) Section(typeID uint32) ([]byte, error) {
	for _, e := range r.TOC {
		if e.TypeID == typeID {
			buf := make([]byte, e.Size)
			if _, err := r.f.ReadAt(buf, int64(e.Offset)); err != nil { return nil, err }
			return buf, nil
		}
	}
	return nil, fmt.Errorf("section %d not found", typeID)
}

// SectionUncompressed returns the raw or decompressed payload depending on flags.
func (r *Reader) SectionUncompressed(typeID uint32) ([]byte, error) {
	for _, e := range r.TOC {
		if e.TypeID != typeID { continue }
		buf := make([]byte, e.Size)
		if _, err := r.f.ReadAt(buf, int64(e.Offset)); err != nil { return nil, err }
		// flags are per-section compression
		if e.Flags&FlagCompZSTD != 0 {
			// zstd decompress
			dec, err := zstdDecode(buf)
			if err != nil { return nil, err }
			return dec, nil
		}
		if e.Flags&FlagCompLZ4 != 0 {
			dec, err := lz4Decode(buf)
			if err != nil { return nil, err }
			return dec, nil
		}
		return buf, nil
	}
	return nil, fmt.Errorf("section %d not found", typeID)
}
