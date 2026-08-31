package events

import (
	"sync"
	"testing"
	"time"
)

func TestSubscribersReceiveTheirDocumentsEvents(t *testing.T) {
	broker := NewBroker()
	stream, cancel := broker.Subscribe(1)
	defer cancel()

	broker.Publish(Event{Type: "page", DocumentID: 1, Message: "one"})

	select {
	case event := <-stream:
		if event.Type != "page" || event.Message != "one" {
			t.Errorf("unexpected event: %+v", event)
		}
		if event.At == 0 {
			t.Error("want a timestamp filled in")
		}
	case <-time.After(time.Second):
		t.Fatal("no event received")
	}
}

func TestEventsAreScopedToOneDocument(t *testing.T) {
	broker := NewBroker()
	stream, cancel := broker.Subscribe(1)
	defer cancel()

	broker.Publish(Event{Type: "page", DocumentID: 2})

	select {
	case event := <-stream:
		t.Errorf("received another document's event: %+v", event)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestEverySubscriberGetsACopy(t *testing.T) {
	broker := NewBroker()
	first, cancelFirst := broker.Subscribe(1)
	second, cancelSecond := broker.Subscribe(1)
	defer cancelFirst()
	defer cancelSecond()

	broker.Publish(Event{Type: "done", DocumentID: 1})

	for i, stream := range []<-chan Event{first, second} {
		select {
		case <-stream:
		case <-time.After(time.Second):
			t.Fatalf("subscriber %d received nothing", i)
		}
	}
}

func TestCancelUnsubscribes(t *testing.T) {
	broker := NewBroker()
	_, cancel := broker.Subscribe(1)
	if broker.SubscriberCount(1) != 1 {
		t.Fatal("subscription not registered")
	}
	cancel()
	if broker.SubscriberCount(1) != 0 {
		t.Error("subscription not removed")
	}
	cancel() // must be safe to call twice
}

func TestSlowSubscriberDoesNotBlockThePublisher(t *testing.T) {
	// A browser on a bad connection must not be able to stall a worker.
	broker := NewBroker()
	_, cancel := broker.Subscribe(1)
	defer cancel()

	done := make(chan struct{})
	go func() {
		for i := 0; i < 10_000; i++ {
			broker.Publish(Event{Type: "page", DocumentID: 1})
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("publishing blocked on a subscriber that never reads")
	}
}

func TestConcurrentSubscribeAndPublish(t *testing.T) {
	broker := NewBroker()
	var wg sync.WaitGroup

	for i := 0; i < 20; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, cancel := broker.Subscribe(1)
			cancel()
		}()
		go func() {
			defer wg.Done()
			broker.Publish(Event{Type: "page", DocumentID: 1})
		}()
	}
	wg.Wait()
}
