from django.shortcuts import render, redirect, get_object_or_404
from .models import Knowledge
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 设置 Embedding Function
embeddings = OpenAIEmbeddings(
    openai_api_key="sk-WKGLTTbN0d583D64555bT3BlBkFJ50E2848daCF9409AAe18",
    openai_api_base="https://cfcus02.opapi.win/v1"
)

# 创建或连接到 Chroma 数据库
vectorstore = Chroma(
    persist_directory="../../../knowledge_base_demo.db",
    embedding_function=embeddings
)

# 查询所有知识
def knowledge_list(request):
    knowledges = Knowledge.objects.all()
    return render(request, 'knowledge_list.html', {'knowledges': knowledges})


# 添加知识
def knowledge_create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        knowledge = Knowledge.objects.create(title=title, content=content)
        vectorstore.add_texts(
            texts=[content],
            metadatas=[{"id": str(knowledge.vector_id)}]
        )
        return redirect('knowledge_list')
    return render(request, 'knowledge_create.html')


# 删除知识
def knowledge_delete(request, knowledge_id):
    knowledge = get_object_or_404(Knowledge, id=knowledge_id)
    vectorID = knowledge.vector_id
    knowledge.delete()
    vectorstore.delete(ids=[str(vectorID)])
    return redirect('knowledge_list')
