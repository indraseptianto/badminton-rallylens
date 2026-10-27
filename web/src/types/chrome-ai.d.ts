interface LanguageModelCreateOptions { initialPrompts?: Array<{ role: "system" | "user" | "assistant"; content: string }>; monitor?: (monitor: EventTarget & { addEventListener(type: "downloadprogress", listener: (event: Event & { loaded: number }) => void): void }) => void }
interface LanguageModelSession { prompt(input: string): Promise<string>; destroy(): void }
interface LanguageModelConstructor { availability(): Promise<"unavailable" | "downloadable" | "downloading" | "available">; create(options?: LanguageModelCreateOptions): Promise<LanguageModelSession> }
declare const LanguageModel: LanguageModelConstructor | undefined;
interface Navigator { gpu?: GPU }
