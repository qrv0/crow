package downloader

import (
	"fmt"
	"io"
	"net/http"
	"os"
)

func Download(url, out string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("http error: %s", resp.Status)
	}
	f, err := os.Create(out)
	if err != nil { return err }
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}
