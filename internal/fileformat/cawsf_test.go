package fileformat

import (
    "bytes"
    "encoding/binary"
    "os"
    "path/filepath"
    "testing"
)

func TestWriterReaderWithCompression(t *testing.T) {
    dir := t.TempDir()
    path := filepath.Join(dir, "test.cawsf")
    // Prepare payloads
    meta := []byte(`{"hello":"world"}`)
    raw := bytes.Repeat([]byte{1,2,3,4}, 1024)
    zst := bytes.Repeat([]byte{5,6,7,8}, 2048)
    // Write
    w := NewWriter()
    w.AddSection(TypeMeta, meta, 0)
    w.AddSection(TypeShardBank, raw, FlagCompLZ4)
    w.AddSection(TypeRouting, zst, FlagCompZSTD)
    if err := w.Write(path); err != nil { t.Fatalf("write error: %v", err) }
    // Read
    r, err := OpenCAWSF(path)
    if err != nil { t.Fatalf("open error: %v", err) }
    defer r.Close()
    // Check magic and toc
    f, _ := os.Open(path)
    defer f.Close()
    head := make([]byte, 8)
    if _, err := f.Read(head); err != nil { t.Fatalf("read head: %v", err) }
    if !bytes.Equal(head, magic[:]) { t.Fatalf("bad magic: %q", string(head)) }
    var hdr struct{ Ver, Num, Res uint32 }
    if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil { t.Fatalf("read hdr: %v", err) }
    if hdr.Num != 3 { t.Fatalf("toc count want 3 got %d", hdr.Num) }
    // Verify decompressed data equals originals
    gotMeta, _ := r.SectionUncompressed(TypeMeta)
    if !bytes.Equal(gotMeta, meta) { t.Fatalf("meta mismatch") }
    gotRaw, _ := r.SectionUncompressed(TypeShardBank)
    if !bytes.Equal(gotRaw, raw) { t.Fatalf("lz4 section mismatch") }
    gotZ, _ := r.SectionUncompressed(TypeRouting)
    if !bytes.Equal(gotZ, zst) { t.Fatalf("zstd section mismatch") }
}

