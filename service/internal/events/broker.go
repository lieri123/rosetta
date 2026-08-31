// Package events is the fan-out behind the progress stream.
//
// One publisher per worker, one subscriber per open browser tab, and no
// storage: an event that nobody is listening for is simply not delivered. The
// authoritative state of a document is in the database and the UI reads it on
// load; this stream exists so that a page being worked on updates without
// polling, not as a record of what happened.
package events

import (
	"sync"
	"time"
)

type Event struct {
	Type       string  `json:"type"` // queued | started | preprocessed | recognised | page | done | failed
	DocumentID int64   `json:"document_id"`
	PageID     int64   `json:"page_id,omitempty"`
	PageIndex  int     `json:"page_index,omitempty"`
	Message    string  `json:"message,omitempty"`
	Completed  int     `json:"completed"`
	Total      int     `json:"total"`
	At         float64 `json:"at"`
}

type subscriber struct {
	channel chan Event
}

type Broker struct {
	mu          sync.RWMutex
	subscribers map[int64]map[*subscriber]struct{}
}

func NewBroker() *Broker {
	return &Broker{subscribers: make(map[int64]map[*subscriber]struct{})}
}

// Subscribe returns a channel of events for one document and a function to
// stop listening. The caller must call the returned function, or the broker
// keeps writing to a channel nobody reads.
func (b *Broker) Subscribe(documentID int64) (<-chan Event, func()) {
	// Buffered because a browser on a slow connection must not be able to
	// stall a worker. If the buffer fills the event is dropped, which is the
	// right trade: progress events are advisory, and the client re-reads real
	// state from the API when the stream ends.
	sub := &subscriber{channel: make(chan Event, 64)}

	b.mu.Lock()
	if b.subscribers[documentID] == nil {
		b.subscribers[documentID] = make(map[*subscriber]struct{})
	}
	b.subscribers[documentID][sub] = struct{}{}
	b.mu.Unlock()

	var once sync.Once
	cancel := func() {
		once.Do(func() {
			b.mu.Lock()
			if set, ok := b.subscribers[documentID]; ok {
				delete(set, sub)
				if len(set) == 0 {
					delete(b.subscribers, documentID)
				}
			}
			b.mu.Unlock()
			close(sub.channel)
		})
	}
	return sub.channel, cancel
}

func (b *Broker) Publish(event Event) {
	if event.At == 0 {
		event.At = float64(time.Now().UnixNano()) / 1e9
	}

	b.mu.RLock()
	defer b.mu.RUnlock()

	for sub := range b.subscribers[event.DocumentID] {
		select {
		case sub.channel <- event:
		default: // slow consumer: drop rather than block the worker
		}
	}
}

// SubscriberCount is used by the tests and by /healthz.
func (b *Broker) SubscriberCount(documentID int64) int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return len(b.subscribers[documentID])
}
