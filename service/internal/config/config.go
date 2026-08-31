// Package config holds the service's settings and where they come from.
//
// Everything has a default that works on a laptop with nothing installed
// except the preprocessing binary, so a fresh checkout runs. Anything that
// would cost money or leak a page off the machine -- the recognition provider
// and its credentials -- has to be asked for explicitly.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type Config struct {
	Addr           string
	DatabasePath   string
	DataDir        string
	StaticDir      string
	PreprocessBin  string
	RescorerURL    string
	Provider       string
	GoogleAPIKey   string
	AzureEndpoint  string
	AzureKey       string
	Workers        int
	RequestTimeout time.Duration
	MaxUploadBytes int64
}

func Default() Config {
	return Config{
		Addr:          "127.0.0.1:8800",
		DatabasePath:  "data/rosetta.db",
		DataDir:       "data",
		StaticDir:     "web/dist",
		PreprocessBin: "preprocess/build/rosetta-preprocess",
		RescorerURL:   "http://127.0.0.1:8801",
		// The mock provider by default: recognition is the only part of this
		// system that costs money per page and sends a page off the machine,
		// so it is opt-in rather than something you discover on your bill.
		Provider: "mock",
		// Recognition is IO-bound on a remote API, so the useful width here is
		// set by the provider's rate limit rather than by cores.
		Workers:        4,
		RequestTimeout: 60 * time.Second,
		MaxUploadBytes: 64 << 20,
	}
}

// FromEnv layers environment variables over the defaults.
func FromEnv() (Config, error) {
	cfg := Default()

	str := func(key string, target *string) {
		if value := os.Getenv(key); value != "" {
			*target = value
		}
	}

	str("ROSETTA_ADDR", &cfg.Addr)
	str("ROSETTA_DB", &cfg.DatabasePath)
	str("ROSETTA_DATA_DIR", &cfg.DataDir)
	str("ROSETTA_STATIC_DIR", &cfg.StaticDir)
	str("ROSETTA_PREPROCESS_BIN", &cfg.PreprocessBin)
	str("ROSETTA_RESCORER_URL", &cfg.RescorerURL)
	str("ROSETTA_PROVIDER", &cfg.Provider)
	str("GOOGLE_VISION_API_KEY", &cfg.GoogleAPIKey)
	str("AZURE_VISION_ENDPOINT", &cfg.AzureEndpoint)
	str("AZURE_VISION_KEY", &cfg.AzureKey)

	if value := os.Getenv("ROSETTA_WORKERS"); value != "" {
		workers, err := strconv.Atoi(value)
		if err != nil || workers < 1 {
			return cfg, fmt.Errorf("ROSETTA_WORKERS: want a positive integer, got %q", value)
		}
		cfg.Workers = workers
	}

	if value := os.Getenv("ROSETTA_TIMEOUT"); value != "" {
		timeout, err := time.ParseDuration(value)
		if err != nil {
			return cfg, fmt.Errorf("ROSETTA_TIMEOUT: %w", err)
		}
		cfg.RequestTimeout = timeout
	}

	return cfg, nil
}

// PageDir is where a document's page images live.
func (c Config) PageDir(documentID int64) string {
	return filepath.Join(c.DataDir, "pages", strconv.FormatInt(documentID, 10))
}

func (c Config) Validate() error {
	if c.Workers < 1 {
		return fmt.Errorf("workers must be at least 1")
	}
	switch c.Provider {
	case "mock", "google", "azure":
	default:
		return fmt.Errorf("unknown provider %q (want mock, google or azure)", c.Provider)
	}
	if c.Provider == "google" && c.GoogleAPIKey == "" {
		return fmt.Errorf("provider google needs GOOGLE_VISION_API_KEY")
	}
	if c.Provider == "azure" && (c.AzureEndpoint == "" || c.AzureKey == "") {
		return fmt.Errorf("provider azure needs AZURE_VISION_ENDPOINT and AZURE_VISION_KEY")
	}
	return nil
}
