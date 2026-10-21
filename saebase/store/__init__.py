"""SAEBase M1: cache activation sites and deterministic SAE codes for reuse.

Start with ActivationIdentity, SampleRecord and ActivationWriter; use
collect_activations() to record new sites and encode_from_store() to reuse those
sites with any compatible SAE without repeating model forwards.
"""
from .identity import (ActivationIdentity, CheckpointLeaseError, CodeIdentity, SampleRecord,
                       TokenSelection, VerifiedCheckpointSession, fingerprint, resolve_positions,
                       sha256_state_dict)
from .shards import (ActivationWriter, CacheIntegrityError, CodeWriter,
                     ShardedReader, StoredSample)
from .collect import (CollectionStats, capture_selected, collect_activations,
                      encode_from_store)

__all__ = ["ActivationIdentity", "CheckpointLeaseError", "CodeIdentity", "SampleRecord",
           "TokenSelection", "VerifiedCheckpointSession", "fingerprint", "resolve_positions",
           "sha256_state_dict", "ActivationWriter", "CodeWriter",
           "ShardedReader", "StoredSample", "CacheIntegrityError", "CollectionStats",
           "capture_selected", "collect_activations", "encode_from_store"]
