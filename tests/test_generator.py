"""
Unit tests for the AnswerGenerator module
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generation.generator import AnswerGenerator


class TestAnswerGenerator:
    """Test suite for AnswerGenerator"""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client"""
        mock = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="This is a test answer"))
        ]
        mock.chat.completions.create.return_value = mock_response
        return mock

    @pytest.fixture
    def mock_streaming_response(self):
        """Mock streaming response from OpenAI"""
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="This "))]),
            Mock(choices=[Mock(delta=Mock(content="is "))]),
            Mock(choices=[Mock(delta=Mock(content="a "))]),
            Mock(choices=[Mock(delta=Mock(content="test"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # End of stream
        ]
        return iter(chunks)

    def test_init_default_model(self):
        """Test initialization with default model"""
        with patch('src.generation.generator.OpenAI'):
            generator = AnswerGenerator()
            assert generator.model == "gpt-4o-mini"

    def test_init_custom_model(self):
        """Test initialization with custom model"""
        with patch('src.generation.generator.OpenAI'):
            generator = AnswerGenerator(model="gpt-4o")
            assert generator.model == "gpt-4o"

    def test_system_prompt_exists(self):
        """Test that system prompt is properly set"""
        with patch('src.generation.generator.OpenAI'):
            generator = AnswerGenerator()
            assert generator.system_prompt is not None
            assert len(generator.system_prompt) > 0
            assert "banking" in generator.system_prompt.lower()
            assert "customer support" in generator.system_prompt.lower()

    @patch('src.generation.generator.OpenAI')
    def test_generate_basic(self, mock_openai_class, mock_openai_client):
        """Test basic answer generation"""
        mock_openai_class.return_value = mock_openai_client

        generator = AnswerGenerator()
        answer = generator.generate(
            query="How do I reset my password?",
            context="[Document 1] You can reset your password in settings."
        )

        assert answer == "This is a test answer"
        mock_openai_client.chat.completions.create.assert_called_once()

    @patch('src.generation.generator.OpenAI')
    def test_generate_with_conversation_history(self, mock_openai_class,
                                                mock_openai_client):
        """Test generation with conversation history"""
        mock_openai_class.return_value = mock_openai_client

        conversation_history = [
            {"role": "user", "content": "What is a transfer fee?"},
            {"role": "assistant", "content": "A transfer fee is..."}
        ]

        generator = AnswerGenerator()
        answer = generator.generate(
            query="How much is it?",
            context="[Document 1] The fee is $5.",
            conversation_history=conversation_history
        )

        assert answer == "This is a test answer"

        # Verify conversation history was included
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']

        # Should have: system prompt + 2 history messages + current query
        assert len(messages) == 4
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[2]['role'] == 'assistant'
        assert messages[3]['role'] == 'user'

    @patch('src.generation.generator.OpenAI')
    def test_generate_limits_conversation_history(self, mock_openai_class,
                                                  mock_openai_client):
        """Test that conversation history is limited to last 6 messages"""
        mock_openai_class.return_value = mock_openai_client

        # Create 10 messages (5 turns)
        long_conversation = [
            {"role": "user", "content": f"Question {i}"}
            if i % 2 == 0 else
            {"role": "assistant", "content": f"Answer {i}"}
            for i in range(10)
        ]

        generator = AnswerGenerator()
        answer = generator.generate(
            query="Current question",
            context="Context",
            conversation_history=long_conversation
        )

        # Verify only last 6 messages from history are included
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']

        # Should have: system + last 6 history + current = 8 total
        assert len(messages) == 8

    @patch('src.generation.generator.OpenAI')
    def test_generate_with_custom_parameters(self, mock_openai_class,
                                            mock_openai_client):
        """Test generation with custom temperature and max_tokens"""
        mock_openai_class.return_value = mock_openai_client

        generator = AnswerGenerator()
        answer = generator.generate(
            query="Test query",
            context="Test context",
            temperature=0.5,
            max_tokens=500
        )

        # Verify custom parameters were passed
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 500

    @patch('src.generation.generator.OpenAI')
    def test_generate_message_format(self, mock_openai_class, mock_openai_client):
        """Test that messages are formatted correctly"""
        mock_openai_class.return_value = mock_openai_client

        generator = AnswerGenerator()
        test_query = "What are the transfer fees?"
        test_context = "[Document 1] Transfer fees are $5."

        generator.generate(query=test_query, context=test_context)

        # Verify message structure
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']

        # Check system message
        assert messages[0]['role'] == 'system'
        assert len(messages[0]['content']) > 0

        # Check user message contains both query and context
        assert messages[1]['role'] == 'user'
        user_content = messages[1]['content']
        assert test_query in user_content
        assert test_context in user_content

    @patch('src.generation.generator.OpenAI')
    def test_generate_stream(self, mock_openai_class, mock_openai_client,
                            mock_streaming_response):
        """Test streaming generation"""
        mock_openai_class.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.return_value = mock_streaming_response

        generator = AnswerGenerator()
        chunks = list(generator.generate_stream(
            query="Test query",
            context="Test context"
        ))

        # Should get 4 chunks (excluding None content)
        assert len(chunks) == 4
        assert chunks == ["This ", "is ", "a ", "test"]

        # Verify streaming was enabled
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]['stream'] is True

    @patch('src.generation.generator.OpenAI')
    def test_generate_stream_with_conversation_history(self, mock_openai_class,
                                                       mock_openai_client,
                                                       mock_streaming_response):
        """Test streaming with conversation history"""
        mock_openai_class.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.return_value = mock_streaming_response

        conversation_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        generator = AnswerGenerator()
        chunks = list(generator.generate_stream(
            query="Current question",
            context="Context",
            conversation_history=conversation_history
        ))

        assert len(chunks) == 4

        # Verify conversation history was included
        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 4  # system + 2 history + current

    @patch('src.generation.generator.OpenAI')
    def test_generate_empty_context(self, mock_openai_class, mock_openai_client):
        """Test generation with empty context"""
        mock_openai_class.return_value = mock_openai_client

        generator = AnswerGenerator()
        answer = generator.generate(
            query="Test query",
            context=""
        )

        assert answer == "This is a test answer"
        # Should still work, just with empty context

    @patch('src.generation.generator.OpenAI')
    def test_generate_uses_correct_model(self, mock_openai_class, mock_openai_client):
        """Test that the specified model is used"""
        mock_openai_class.return_value = mock_openai_client

        custom_model = "gpt-4o"
        generator = AnswerGenerator(model=custom_model)
        generator.generate(query="Test", context="Context")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]['model'] == custom_model

    @patch('src.generation.generator.OpenAI')
    def test_generate_stream_concatenation(self, mock_openai_class,
                                          mock_openai_client, mock_streaming_response):
        """Test that streaming chunks can be concatenated to full answer"""
        mock_openai_class.return_value = mock_openai_client
        mock_openai_client.chat.completions.create.return_value = mock_streaming_response

        generator = AnswerGenerator()
        chunks = list(generator.generate_stream(
            query="Test query",
            context="Test context"
        ))

        full_answer = "".join(chunks)
        assert full_answer == "This is a test"

    @patch('src.generation.generator.OpenAI')
    def test_generate_default_parameters(self, mock_openai_class, mock_openai_client):
        """Test that default parameters are applied correctly"""
        mock_openai_class.return_value = mock_openai_client

        generator = AnswerGenerator()
        generator.generate(query="Test", context="Context")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 300
