// Command rosetta is the handwriting-to-text service.
//
// It cleans page images with the C preprocessor, sends them to a recognition
// provider, infers layout from the word boxes, rescores the result through the
// Python noisy-channel model, and serves the text along with the confidence
// spans the editor underlines.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/lieri123/rosetta/service/internal/api"
	"github.com/lieri123/rosetta/service/internal/config"
	"github.com/lieri123/rosetta/service/internal/events"
	"github.com/lieri123/rosetta/service/internal/pipeline"
	"github.com/lieri123/rosetta/service/internal/preprocess"
	"github.com/lieri123/rosetta/service/internal/queue"
	"github.com/lieri123/rosetta/service/internal/recognize"
	"github.com/lieri123/rosetta/service/internal/rescore"
	"github.com/lieri123/rosetta/service/internal/store"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "rosetta:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.FromEnv()
	if err != nil {
		return err
	}

	flag.StringVar(&cfg.Addr, "addr", cfg.Addr, "address to listen on")
	flag.StringVar(&cfg.DatabasePath, "db", cfg.DatabasePath, "SQLite database path")
	flag.StringVar(&cfg.DataDir, "data", cfg.DataDir, "directory for page images")
	flag.StringVar(&cfg.StaticDir, "static", cfg.StaticDir, "directory of built web assets")
	flag.StringVar(&cfg.Provider, "provider", cfg.Provider, "recognition provider: mock, google or azure")
	flag.StringVar(&cfg.PreprocessBin, "preprocess", cfg.PreprocessBin, "path to rosetta-preprocess")
	flag.StringVar(&cfg.RescorerURL, "rescorer", cfg.RescorerURL, "base URL of the Python rescorer")
	flag.IntVar(&cfg.Workers, "workers", cfg.Workers, "number of page workers")
	mockFixture := flag.String("mock-fixture", "", "replay a recorded provider response (mock provider only)")
	flag.Parse()

	if err := cfg.Validate(); err != nil {
		return err
	}

	logger := log.New(os.Stderr, "", log.LstdFlags|log.Lmsgprefix)

	database, err := store.Open(cfg.DatabasePath)
	if err != nil {
		return err
	}
	defer database.Close()

	provider, err := buildProvider(cfg, *mockFixture)
	if err != nil {
		return err
	}

	runner := preprocess.NewRunner(cfg.PreprocessBin)
	if !runner.Available() {
		logger.Printf("warning: preprocessor not found at %s; pages will be recognised unprocessed "+
			"(run `make preprocess`)", cfg.PreprocessBin)
	}

	rescorer := rescore.New(cfg.RescorerURL)
	broker := events.NewBroker()

	processor := &pipeline.Processor{
		Store: database, Preprocess: runner, Provider: provider,
		Rescorer: rescorer, Broker: broker, DataDir: cfg.DataDir, Logger: logger,
	}

	pool := &queue.Pool{
		Store: database, Handler: processor.ProcessPage, Broker: broker,
		Workers: cfg.Workers, Logger: logger,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool.Start(ctx)

	server := &api.Server{
		Config: cfg, Store: database, Broker: broker, Pool: pool,
		Rescorer: rescorer, Provider: provider.Name(), Logger: logger,
	}

	httpServer := &http.Server{
		Addr:              cfg.Addr,
		Handler:           server.Routes(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: the progress stream is a long-lived response, and a
		// write deadline would cut it off mid-document.
		IdleTimeout: 120 * time.Second,
	}

	go func() {
		<-ctx.Done()
		logger.Println("shutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		httpServer.Shutdown(shutdownCtx)
	}()

	logger.Printf("listening on http://%s", cfg.Addr)
	logger.Printf("provider %s, %d worker(s), database %s", provider.Name(), cfg.Workers, cfg.DatabasePath)
	if rescorer.Healthy(ctx) {
		logger.Printf("rescorer reachable at %s", cfg.RescorerURL)
	} else {
		logger.Printf("warning: rescorer unreachable at %s; tiering will fall back to "+
			"recognition confidence alone", cfg.RescorerURL)
	}

	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}

	pool.Wait()
	return nil
}

func buildProvider(cfg config.Config, fixture string) (recognize.Provider, error) {
	if fixture != "" {
		if cfg.Provider != "mock" {
			return nil, fmt.Errorf("-mock-fixture only applies to the mock provider")
		}
		return recognize.NewMockFromFile(fixture)
	}
	return recognize.New(cfg.Provider, cfg.GoogleAPIKey, cfg.AzureEndpoint, cfg.AzureKey)
}
