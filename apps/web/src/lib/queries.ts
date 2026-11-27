"use client";

import { useQuery } from "@tanstack/react-query";

import { api, unwrap } from "@/lib/api/client";
import { ACTIVE_DATASET_STATUSES } from "@/lib/format";

const POLL_MS = 2000;
const ACTIVE_JOB = new Set(["queued", "running", "cancel_requested"]);

export function useDataset(datasetId: string) {
  return useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/datasets/{dataset_id}", {
          params: { path: { dataset_id: datasetId } },
        }),
      ),
    refetchInterval: (query) =>
      query.state.data && ACTIVE_DATASET_STATUSES.has(query.state.data.status) ? POLL_MS : false,
  });
}

export function useVersion(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}", {
          params: { path: { version_id: versionId! } },
        }),
      ),
    enabled: Boolean(versionId),
  });
}

export function usePreview(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "preview"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/preview", {
          params: { path: { version_id: versionId! } },
        }),
      ),
    enabled: Boolean(versionId),
  });
}

/** Latest stored profile plus the latest profiling job; polls while that job is active. */
export function useProfileState(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "profile"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/profile", {
          params: { path: { version_id: versionId! } },
        }),
      ),
    enabled: Boolean(versionId),
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

export function isActiveJob(status: string | undefined): boolean {
  return status !== undefined && ACTIVE_JOB.has(status);
}

export function useDomains() {
  return useQuery({
    queryKey: ["domains"],
    queryFn: async () => unwrap(await api.GET("/api/v1/domains")).items,
    staleTime: 5 * 60_000,
  });
}

export function useRoles(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "roles"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/roles", {
          params: { path: { version_id: versionId! } },
        }),
      ),
    enabled: Boolean(versionId),
  });
}

/** Latest diagnostics run plus the latest analysis job; polls while that job is active. */
export function useAnalysis(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "analysis"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/analyses/latest", {
          params: { path: { version_id: versionId! } },
        }),
      ),
    enabled: Boolean(versionId),
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

/** The preprocessing op catalogue (static per deployment). */
export function useOps() {
  return useQuery({
    queryKey: ["ops"],
    queryFn: async () => unwrap(await api.GET("/api/v1/ops")).items,
    staleTime: 5 * 60_000,
  });
}

/** Latest recommendation set for a version (generated on request, never automatically). */
export function useRecommendations(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "recommendations"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/recommendations", {
          params: { path: { version_id: versionId! } },
        }),
      ).set,
    enabled: Boolean(versionId),
  });
}

export function useExperiments(versionId: string | undefined) {
  return useQuery({
    queryKey: ["versions", versionId, "experiments"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/versions/{version_id}/experiments", {
          params: { path: { version_id: versionId! } },
        }),
      ).items,
    enabled: Boolean(versionId),
    refetchInterval: (query) =>
      query.state.data?.some((e) => e.job && ACTIVE_JOB.has(e.job.status)) ? POLL_MS : false,
  });
}

/** One experiment with its results; polls while its job is active. */
export function useExperiment(experimentId: string | null) {
  return useQuery({
    queryKey: ["experiments", experimentId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/experiments/{experiment_id}", {
          params: { path: { experiment_id: experimentId! } },
        }),
      ),
    enabled: Boolean(experimentId),
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

/** A pipeline; polls while its impact preview job is active. */
export function usePipeline(pipelineId: string) {
  return useQuery({
    queryKey: ["pipelines", pipelineId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/pipelines/{pipeline_id}", {
          params: { path: { pipeline_id: pipelineId } },
        }),
      ),
    refetchInterval: (query) => {
      const job = query.state.data?.preview_job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

export function useImpactPreview(pipelineId: string, version: string | undefined) {
  return useQuery({
    queryKey: ["pipelines", pipelineId, "preview", version],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/pipelines/{pipeline_id}/impact-preview", {
          params: { path: { pipeline_id: pipelineId } },
        }),
      ),
  });
}

export function usePipelines(datasetId: string) {
  return useQuery({
    queryKey: ["datasets", datasetId, "pipelines"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/datasets/{dataset_id}/pipelines", {
          params: { path: { dataset_id: datasetId } },
        }),
      ).items,
  });
}

export function useVersionList(datasetId: string) {
  return useQuery({
    queryKey: ["datasets", datasetId, "versions"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/datasets/{dataset_id}/versions", {
          params: { path: { dataset_id: datasetId } },
        }),
      ).items,
  });
}

export function useTransformationRun(runId: string | null | undefined) {
  return useQuery({
    queryKey: ["transformation-runs", runId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/transformation-runs/{transformation_run_id}", {
          params: { path: { transformation_run_id: runId! } },
        }),
      ),
    enabled: Boolean(runId),
  });
}

/** A job; polls until it finishes. */
export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ["jobs", jobId],
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/jobs/{job_id}", { params: { path: { job_id: jobId! } } })),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      query.state.data && ACTIVE_JOB.has(query.state.data.status) ? POLL_MS : false,
  });
}

/** A version comparison result; polls while its job runs. */
export function useComparison(jobId: string | null) {
  return useQuery({
    queryKey: ["version-comparisons", jobId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/version-comparisons/{job_id}", {
          params: { path: { job_id: jobId! } },
        }),
      ),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

export function useModelCatalogue() {
  return useQuery({
    queryKey: ["ml", "models"],
    queryFn: async () => unwrap(await api.GET("/api/v1/ml/models")),
    staleTime: Infinity,
  });
}

/** Training runs of a dataset; polls while any run's job is active. */
export function useTrainingRuns(datasetId: string) {
  return useQuery({
    queryKey: ["datasets", datasetId, "training-runs"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/datasets/{dataset_id}/training-runs", {
          params: { path: { dataset_id: datasetId } },
        }),
      ).items,
    refetchInterval: (query) =>
      query.state.data?.some((r) => r.job && ACTIVE_JOB.has(r.job.status)) ? POLL_MS : false,
  });
}

/** One training run with its model evaluations; polls while its job is active. */
export function useTrainingRun(runId: string) {
  return useQuery({
    queryKey: ["training-runs", runId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/training-runs/{training_run_id}", {
          params: { path: { training_run_id: runId } },
        }),
      ),
    refetchInterval: (query) => {
      const job = query.state.data?.job;
      return job && ACTIVE_JOB.has(job.status) ? POLL_MS : false;
    },
  });
}

/** The latest reliability audit of a run; polls until it exists and its job finished. */
export function useTrainingAudit(runId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["training-runs", runId, "audit"],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/training-runs/{training_run_id}/audit", {
          params: { path: { training_run_id: runId } },
        }),
      ),
    enabled,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return POLL_MS;
      return !data.run || (data.job && ACTIVE_JOB.has(data.job.status)) ? POLL_MS : false;
    },
  });
}
