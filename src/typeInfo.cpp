#include "typeInfo.h"
#include "typeinfo.hpp"

const TypeInfo* getTypeInfo(ObjectHandle obj)
{
    return &obj->getTypeInfo();
}

const bool isParent(const TypeInfo* obj, const TypeInfo* parent)
{
    while (obj != nullptr)
    {
        if (obj == parent)
        {
            return true;
        }
        obj = obj->parent;
    }

    return false;
}

const TypeInfo& Object::getStaticTypeInfo()
{
    static TypeInfo typeInfo{nullptr};
    return typeInfo; 
}

const TypeInfo& Object::getTypeInfo()
{
    return getStaticTypeInfo(); 
}
