package main

import "testing"

func TestRankByCosineOrder(t *testing.T) {
    // two keys: [1,0] and [0,1], query [1,0] should pick idx 0 first
    keys := [][]float32{{1,0},{0,1}}
    q := []float32{1,0}
    order := rankByCosine(keys, q)
    if len(order) != 2 || order[0] != 0 { t.Fatalf("unexpected order: %v", order) }
}

