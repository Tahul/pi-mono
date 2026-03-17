import { clearApiProviders, registerApiProvider } from "../api-registry.js";
import type {
	Api,
	AssistantMessage,
	AssistantMessageEvent,
	Context,
	Model,
	SimpleStreamOptions,
	StreamOptions,
} from "../types.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";

// ---------------------------------------------------------------------------
// Generic lazy-loading infrastructure
// ---------------------------------------------------------------------------

type DynamicImport = (specifier: string) => Promise<unknown>;
const dynamicImport: DynamicImport = (specifier) => import(specifier);

/**
 * Forward events from an async iterable into an AssistantMessageEventStream.
 */
function forwardStream(target: AssistantMessageEventStream, source: AsyncIterable<AssistantMessageEvent>): void {
	(async () => {
		for await (const event of source) {
			target.push(event);
		}
		target.end();
	})();
}

/**
 * Create a stub error message when a provider module fails to load.
 */
function createLazyLoadErrorMessage<TApi extends Api>(api: TApi, model: Model<TApi>, error: unknown): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api,
		provider: model.provider,
		model: model.id,
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "error",
		errorMessage: error instanceof Error ? error.message : String(error),
		timestamp: Date.now(),
	};
}

/**
 * Descriptor for a lazily loaded provider module.
 */
interface LazyProviderDescriptor<TApi extends Api> {
	api: TApi;
	/** Module specifier for dynamic import (string-concatenated to prevent bundler inlining). */
	specifier: string;
	/** Name of the stream export in the module. */
	streamExport: string;
	/** Name of the streamSimple export in the module. */
	streamSimpleExport: string;
	/** Human-readable SDK package name shown in error messages. */
	sdkPackage: string;
}

/**
 * Register a lazily loaded provider. The provider module is only imported when
 * a model using its API is first streamed. If the SDK package is not installed,
 * the user gets a clear error message.
 */
function registerLazyProvider<TApi extends Api>(descriptor: LazyProviderDescriptor<TApi>): void {
	const { api, specifier, streamExport, streamSimpleExport, sdkPackage } = descriptor;

	function streamLazy(model: Model<TApi>, context: Context, options?: StreamOptions): AssistantMessageEventStream {
		const outer = new AssistantMessageEventStream();

		dynamicImport(specifier)
			.then((module: any) => {
				const inner = module[streamExport](model, context, options);
				forwardStream(outer, inner);
			})
			.catch((error) => {
				const message = createLazyLoadErrorMessage(api, model, error);
				const errorMsg =
					error instanceof Error && error.message.includes("Cannot find")
						? `Provider SDK not available. Install ${sdkPackage} to use ${api} models.`
						: error instanceof Error
							? error.message
							: String(error);
				const errorMessage = { ...message, errorMessage: errorMsg };
				outer.push({ type: "error", reason: "error", error: errorMessage });
				outer.end(errorMessage);
			});

		return outer;
	}

	function streamSimpleLazy(
		model: Model<TApi>,
		context: Context,
		options?: SimpleStreamOptions,
	): AssistantMessageEventStream {
		const outer = new AssistantMessageEventStream();

		dynamicImport(specifier)
			.then((module: any) => {
				const inner = module[streamSimpleExport](model, context, options);
				forwardStream(outer, inner);
			})
			.catch((error) => {
				const message = createLazyLoadErrorMessage(api, model, error);
				const errorMsg =
					error instanceof Error && error.message.includes("Cannot find")
						? `Provider SDK not available. Install ${sdkPackage} to use ${api} models.`
						: error instanceof Error
							? error.message
							: String(error);
				const errorMessage = { ...message, errorMessage: errorMsg };
				outer.push({ type: "error", reason: "error", error: errorMessage });
				outer.end(errorMessage);
			});

		return outer;
	}

	registerApiProvider({
		api,
		stream: streamLazy as any,
		streamSimple: streamSimpleLazy as any,
	});
}

// ---------------------------------------------------------------------------
// Bedrock-specific lazy loading (preserves the override mechanism)
// ---------------------------------------------------------------------------

interface BedrockProviderModule {
	streamBedrock: (
		model: Model<"bedrock-converse-stream">,
		context: Context,
		options?: StreamOptions,
	) => AsyncIterable<AssistantMessageEvent>;
	streamSimpleBedrock: (
		model: Model<"bedrock-converse-stream">,
		context: Context,
		options?: SimpleStreamOptions,
	) => AsyncIterable<AssistantMessageEvent>;
}

const BEDROCK_PROVIDER_SPECIFIER = "./amazon-" + "bedrock.js";

let bedrockProviderModuleOverride: BedrockProviderModule | undefined;

export function setBedrockProviderModule(module: BedrockProviderModule): void {
	bedrockProviderModuleOverride = module;
}

async function loadBedrockProviderModule(): Promise<BedrockProviderModule> {
	if (bedrockProviderModuleOverride) {
		return bedrockProviderModuleOverride;
	}
	const module = await dynamicImport(BEDROCK_PROVIDER_SPECIFIER);
	return module as BedrockProviderModule;
}

function streamBedrockLazy(
	model: Model<"bedrock-converse-stream">,
	context: Context,
	options?: StreamOptions,
): AssistantMessageEventStream {
	const outer = new AssistantMessageEventStream();

	loadBedrockProviderModule()
		.then((module) => {
			const inner = module.streamBedrock(model, context, options);
			forwardStream(outer, inner);
		})
		.catch((error) => {
			const message = createLazyLoadErrorMessage("bedrock-converse-stream", model, error);
			outer.push({ type: "error", reason: "error", error: message });
			outer.end(message);
		});

	return outer;
}

function streamSimpleBedrockLazy(
	model: Model<"bedrock-converse-stream">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const outer = new AssistantMessageEventStream();

	loadBedrockProviderModule()
		.then((module) => {
			const inner = module.streamSimpleBedrock(model, context, options);
			forwardStream(outer, inner);
		})
		.catch((error) => {
			const message = createLazyLoadErrorMessage("bedrock-converse-stream", model, error);
			outer.push({ type: "error", reason: "error", error: message });
			outer.end(message);
		});

	return outer;
}

// ---------------------------------------------------------------------------
// Provider specifiers (string-concatenated to prevent static analysis / bundler inlining)
// ---------------------------------------------------------------------------

const ANTHROPIC_SPECIFIER = "./anthro" + "pic.js";
const OPENAI_COMPLETIONS_SPECIFIER = "./openai-comple" + "tions.js";
const OPENAI_RESPONSES_SPECIFIER = "./openai-respon" + "ses.js";
const AZURE_OPENAI_RESPONSES_SPECIFIER = "./azure-openai-respon" + "ses.js";
const OPENAI_CODEX_RESPONSES_SPECIFIER = "./openai-codex-respon" + "ses.js";
const GOOGLE_SPECIFIER = "./goo" + "gle.js";
const GOOGLE_GEMINI_CLI_SPECIFIER = "./google-gemini-" + "cli.js";
const GOOGLE_VERTEX_SPECIFIER = "./google-ver" + "tex.js";
const MISTRAL_SPECIFIER = "./mis" + "tral.js";

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

export function registerBuiltInApiProviders(): void {
	registerLazyProvider({
		api: "anthropic-messages",
		specifier: ANTHROPIC_SPECIFIER,
		streamExport: "streamAnthropic",
		streamSimpleExport: "streamSimpleAnthropic",
		sdkPackage: "@anthropic-ai/sdk",
	});

	registerLazyProvider({
		api: "openai-completions",
		specifier: OPENAI_COMPLETIONS_SPECIFIER,
		streamExport: "streamOpenAICompletions",
		streamSimpleExport: "streamSimpleOpenAICompletions",
		sdkPackage: "openai",
	});

	registerLazyProvider({
		api: "mistral-conversations",
		specifier: MISTRAL_SPECIFIER,
		streamExport: "streamMistral",
		streamSimpleExport: "streamSimpleMistral",
		sdkPackage: "@mistralai/mistralai",
	});

	registerLazyProvider({
		api: "openai-responses",
		specifier: OPENAI_RESPONSES_SPECIFIER,
		streamExport: "streamOpenAIResponses",
		streamSimpleExport: "streamSimpleOpenAIResponses",
		sdkPackage: "openai",
	});

	registerLazyProvider({
		api: "azure-openai-responses",
		specifier: AZURE_OPENAI_RESPONSES_SPECIFIER,
		streamExport: "streamAzureOpenAIResponses",
		streamSimpleExport: "streamSimpleAzureOpenAIResponses",
		sdkPackage: "openai",
	});

	registerLazyProvider({
		api: "openai-codex-responses",
		specifier: OPENAI_CODEX_RESPONSES_SPECIFIER,
		streamExport: "streamOpenAICodexResponses",
		streamSimpleExport: "streamSimpleOpenAICodexResponses",
		sdkPackage: "openai",
	});

	registerLazyProvider({
		api: "google-generative-ai",
		specifier: GOOGLE_SPECIFIER,
		streamExport: "streamGoogle",
		streamSimpleExport: "streamSimpleGoogle",
		sdkPackage: "@google/genai",
	});

	registerLazyProvider({
		api: "google-gemini-cli",
		specifier: GOOGLE_GEMINI_CLI_SPECIFIER,
		streamExport: "streamGoogleGeminiCli",
		streamSimpleExport: "streamSimpleGoogleGeminiCli",
		sdkPackage: "@google/genai",
	});

	registerLazyProvider({
		api: "google-vertex",
		specifier: GOOGLE_VERTEX_SPECIFIER,
		streamExport: "streamGoogleVertex",
		streamSimpleExport: "streamSimpleGoogleVertex",
		sdkPackage: "@google/genai",
	});

	// Bedrock keeps its own lazy loading with override support
	registerApiProvider({
		api: "bedrock-converse-stream",
		stream: streamBedrockLazy,
		streamSimple: streamSimpleBedrockLazy,
	});
}

export function resetApiProviders(): void {
	clearApiProviders();
	registerBuiltInApiProviders();
}

registerBuiltInApiProviders();
