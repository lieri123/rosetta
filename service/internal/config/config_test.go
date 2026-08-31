package config

import (
	"strings"
	"testing"
)

func TestDefaultsAreSafe(t *testing.T) {
	cfg := Default()
	// Recognition is the only step that costs money per page and sends ink to
	// someone else's computer. It has to be asked for, not discovered.
	if cfg.Provider != "mock" {
		t.Errorf("want the mock provider by default, got %q", cfg.Provider)
	}
	if !strings.HasPrefix(cfg.Addr, "127.0.0.1") {
		t.Errorf("want a loopback default address, got %q", cfg.Addr)
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("the defaults do not validate: %v", err)
	}
}

func TestValidateRequiresCredentialsForPaidProviders(t *testing.T) {
	cfg := Default()
	cfg.Provider = "google"
	if err := cfg.Validate(); err == nil {
		t.Error("want an error when the Vision key is missing")
	}
	cfg.GoogleAPIKey = "key"
	if err := cfg.Validate(); err != nil {
		t.Errorf("want google to validate with a key: %v", err)
	}

	cfg = Default()
	cfg.Provider = "azure"
	if err := cfg.Validate(); err == nil {
		t.Error("want an error when the Azure endpoint and key are missing")
	}
}

func TestValidateRejectsUnknownProviders(t *testing.T) {
	cfg := Default()
	cfg.Provider = "hieroglyphs"
	if err := cfg.Validate(); err == nil {
		t.Error("want an error for an unknown provider")
	}
}

func TestValidateRejectsZeroWorkers(t *testing.T) {
	cfg := Default()
	cfg.Workers = 0
	if err := cfg.Validate(); err == nil {
		t.Error("want an error for a pool with no workers")
	}
}

func TestFromEnvLayersOverDefaults(t *testing.T) {
	t.Setenv("ROSETTA_ADDR", "0.0.0.0:9000")
	t.Setenv("ROSETTA_WORKERS", "9")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Addr != "0.0.0.0:9000" || cfg.Workers != 9 {
		t.Errorf("environment not applied: %+v", cfg)
	}
	if cfg.Provider != Default().Provider {
		t.Errorf("unset variables should keep their defaults, got %q", cfg.Provider)
	}
}

func TestFromEnvRejectsNonsense(t *testing.T) {
	t.Setenv("ROSETTA_WORKERS", "many")
	if _, err := FromEnv(); err == nil {
		t.Error("want an error for a non-numeric worker count")
	}
}

func TestPageDirIsPerDocument(t *testing.T) {
	cfg := Default()
	if cfg.PageDir(1) == cfg.PageDir(2) {
		t.Error("two documents must not share a page directory")
	}
}
