package quant

import (
	"math/rand"
)

type PQ struct {
	M     int
	K     int
	Dsub  int
	Codebooks [][]float32 // [m][k*dsub]
}

// TrainPQ trains a Product Quantizer on data (N x D), with D divisible by m.
func TrainPQ(data [][]float32, m, k int, iters int, seed int64) *PQ {
	N := len(data)
	if N == 0 { return &PQ{M: m, K: k} }
	D := len(data[0])
	dsub := D / m
	// ensure k <= N to avoid degenerate kmeans init
	if k > N { k = N }
	if k < 1 { k = 1 }
	pq := &PQ{M: m, K: k, Dsub: dsub, Codebooks: make([][]float32, m)}
	rng := rand.New(rand.NewSource(seed))
	// simple KMeans per subvector
	for i := 0; i < m; i++ {
		// extract sub-vectors
		subs := make([][]float32, N)
		for n := 0; n < N; n++ { subs[n] = data[n][i*dsub:(i+1)*dsub] }
		pq.Codebooks[i] = kmeans(subs, k, iters, rng)
	}
	return pq
}

func kmeans(data [][]float32, k, iters int, rng *rand.Rand) []float32 {
	N := len(data)
	D := len(data[0])
	centroids := make([][]float32, k)
	idx := rng.Perm(N)[:k]
	for i := 0; i < k; i++ {
		t := make([]float32, D)
		copy(t, data[idx[i]])
		centroids[i] = t
	}
	assign := make([]int, N)
	for it := 0; it < iters; it++ {
		// assign step
		for n := 0; n < N; n++ {
			best, bestd := 0, float32(1e30)
			for j := 0; j < k; j++ {
				d := l2(data[n], centroids[j])
				if d < bestd { bestd, best = d, j }
			}
			assign[n] = best
		}
		// update
		counts := make([]int, k)
		sums := make([][]float32, k)
		for j := 0; j < k; j++ { sums[j] = make([]float32, D) }
		for n := 0; n < N; n++ {
			c := assign[n]
			counts[c]++
			accumAdd(sums[c], data[n])
		}
		for j := 0; j < k; j++ {
			if counts[j] == 0 { continue }
			inv := 1.0 / float32(counts[j])
			for d := 0; d < D; d++ { sums[j][d] *= float32(inv) }
			centroids[j] = sums[j]
		}
	}
	// flatten
	flat := make([]float32, k*D)
	for j := 0; j < k; j++ { copy(flat[j*D:(j+1)*D], centroids[j]) }
	return flat
}

func l2(a, b []float32) float32 {
	s := float32(0)
	for i := range a { d := a[i]-b[i]; s += d*d }
	return s
}

func accumAdd(dst, src []float32) {
	for i := range dst { dst[i] += src[i] }
}

// Encode returns (N x m) codes
func (pq *PQ) Encode(data [][]float32) [][]uint8 {
	N := len(data)
	_ = len(data[0])
	codes := make([][]uint8, N)
	for n := 0; n < N; n++ { codes[n] = make([]uint8, pq.M) }
	for i := 0; i < pq.M; i++ {
		dsub := pq.Dsub
		cb := pq.Codebooks[i]
		for n := 0; n < N; n++ {
			best, bestd := 0, float32(1e30)
			for j := 0; j < pq.K; j++ {
				start := j*dsub
				d := l2Flat(data[n][i*dsub:(i+1)*dsub], cb[start:start+dsub])
				if d < bestd { bestd, best = d, j }
			}
			codes[n][i] = uint8(best)
		}
	}
	return codes
}

func l2Flat(a []float32, b []float32) float32 {
	s := float32(0)
	for i := range a { d := a[i]-b[i]; s += d*d }
	return s
}

// Decode reconstructs data from codes
func (pq *PQ) Decode(codes [][]uint8) [][]float32 {
	N := len(codes)
	D := pq.M * pq.Dsub
	out := make([][]float32, N)
	for n := 0; n < N; n++ { out[n] = make([]float32, D) }
	for i := 0; i < pq.M; i++ {
		dsub := pq.Dsub
		cb := pq.Codebooks[i]
		for n := 0; n < N; n++ {
			idx := int(codes[n][i])
			start := idx*dsub
			copy(out[n][i*dsub:(i+1)*dsub], cb[start:start+dsub])
		}
	}
	return out
}
