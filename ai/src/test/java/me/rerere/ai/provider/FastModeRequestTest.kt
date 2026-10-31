package me.rerere.ai.provider

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import me.rerere.ai.provider.providers.claude.ClaudeProvider
import me.rerere.ai.provider.providers.claude.messageRequestHeaders
import me.rerere.ai.provider.providers.openai.ChatCompletionsAPI
import me.rerere.ai.provider.providers.openai.ResponseAPI
import me.rerere.ai.provider.providers.openai.resolveResponseProviderCapabilities
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.util.KeyRoulette
import okhttp3.OkHttpClient
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FastModeRequestTest {
    private val messages = listOf(UIMessage.user("hello"))

    private fun params(modelId: String, fastMode: Boolean) = TextGenerationParams(
        model = Model(modelId = modelId),
        fastMode = fastMode,
    )

    private fun claudeRequest(params: TextGenerationParams): JsonObject {
        val method = ClaudeProvider::class.java.getDeclaredMethod(
            "buildMessageRequest",
            ProviderSetting.Claude::class.java,
            List::class.java,
            TextGenerationParams::class.java,
            Boolean::class.javaPrimitiveType!!
        )
        method.isAccessible = true
        return method.invoke(
            ClaudeProvider(OkHttpClient()),
            ProviderSetting.Claude(),
            messages,
            params,
            true
        ) as JsonObject
    }

    private fun chatCompletionsRequest(params: TextGenerationParams): JsonObject {
        val method = ChatCompletionsAPI::class.java.getDeclaredMethod(
            "buildChatCompletionRequest",
            List::class.java,
            TextGenerationParams::class.java,
            ProviderSetting.OpenAI::class.java,
            Boolean::class.javaPrimitiveType
        )
        method.isAccessible = true
        val api = ChatCompletionsAPI(OkHttpClient(), KeyRoulette.default())
        return method.invoke(api, messages, params, ProviderSetting.OpenAI(), true) as JsonObject
    }

    private fun responsesRequest(params: TextGenerationParams): JsonObject =
        ResponseAPI(OkHttpClient()).buildRequestBody(
            capabilities = resolveResponseProviderCapabilities("api.openai.com"),
            messages = messages,
            params = params,
            stream = true,
        )

    @Test
    fun `claude fast mode is limited to supported opus models`() {
        val claude = ProviderSetting.Claude()
        assertTrue(claude.supportsFastMode(Model(modelId = "claude-opus-5-5")))
        assertTrue(claude.supportsFastMode(Model(modelId = "claude-opus-5")))
        assertTrue(claude.supportsFastMode(Model(modelId = "claude-opus-4-8")))
        assertTrue(claude.supportsFastMode(Model(modelId = "anthropic/claude-opus-4.8")))
        assertFalse(claude.supportsFastMode(Model(modelId = "claude-opus-4-7")))
        assertFalse(claude.supportsFastMode(Model(modelId = "claude-opus-4-5-20251101")))
        assertFalse(claude.supportsFastMode(Model(modelId = "claude-sonnet-5-5")))
        assertFalse(claude.supportsFastMode(Model(modelId = "deepseek-v4-pro")))
    }

    @Test
    fun `openai fast mode is limited to gpt models`() {
        val openai = ProviderSetting.OpenAI()
        val chatgpt = ProviderSetting.ChatGPT()
        assertTrue(openai.supportsFastMode(Model(modelId = "gpt-5.5")))
        assertTrue(openai.supportsFastMode(Model(modelId = "gpt-6.1-sol")))
        assertTrue(openai.supportsFastMode(Model(modelId = "gpt-4.1-mini")))
        assertTrue(openai.supportsFastMode(Model(modelId = "o4-mini")))
        assertTrue(chatgpt.supportsFastMode(Model(modelId = "gpt-5.4")))
        assertFalse(openai.supportsFastMode(Model(modelId = "gpt-oss-120b")))
        assertFalse(openai.supportsFastMode(Model(modelId = "deepseek-v4-pro")))
        // GPT 以外的协议不支持
        assertFalse(ProviderSetting.Google().supportsFastMode(Model(modelId = "gpt-5.5")))
        assertFalse(ProviderSetting.Grok().supportsFastMode(Model(modelId = "grok-4")))
    }

    @Test
    fun `claude fast mode sends speed and beta header`() {
        val params = params("claude-opus-5-5", fastMode = true)

        assertEquals("fast", claudeRequest(params)["speed"]?.jsonPrimitive?.content)
        assertEquals("fast-mode-2026-02-01", ProviderSetting.Claude().messageRequestHeaders(params)["anthropic-beta"])
    }

    @Test
    fun `claude fast mode keeps custom beta header`() {
        val params = params("claude-opus-5-5", fastMode = true).copy(
            customHeaders = listOf(CustomHeader("anthropic-beta", "context-1m-2025-08-07")),
        )

        val headers = ProviderSetting.Claude().messageRequestHeaders(params)

        assertEquals(listOf("context-1m-2025-08-07,fast-mode-2026-02-01"), headers.values("anthropic-beta"))
    }

    @Test
    fun `claude standard speed sends neither speed nor beta header`() {
        val params = params("claude-opus-5-5", fastMode = false)

        assertNull(claudeRequest(params)["speed"])
        assertNull(ProviderSetting.Claude().messageRequestHeaders(params)["anthropic-beta"])
    }

    @Test
    fun `openai fast mode sends priority service tier`() {
        val params = params("gpt-5.5", fastMode = true)

        assertEquals("priority", chatCompletionsRequest(params)["service_tier"]?.jsonPrimitive?.content)
        assertEquals("priority", responsesRequest(params)["service_tier"]?.jsonPrimitive?.content)
    }

    @Test
    fun `openai standard speed omits service tier`() {
        val params = params("gpt-5.5", fastMode = false)

        assertNull(chatCompletionsRequest(params)["service_tier"])
        assertNull(responsesRequest(params)["service_tier"])
    }
}
