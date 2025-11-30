type Event = {
    id: string;
    timestamp: number;
    type: "tool_checkin" | "tool_checkout",
    user: {
        id: string;
        name: string;
        email: string;
        imageUrl: string
    }
    tool: {
        id: string;
        name: string;
        description: string;
        imageUrl: string;
        type: string
        cost: number
    },
    eventImageUrl: string
}

type SystemOverview = {
    toolsCount: number;
    usersWithCheckedOutToolsCount: number;
    toolsUnseenInLast7DaysCount: number;
}

export const dummySystemOverview: SystemOverview = {
    toolsCount: 143,
    usersWithCheckedOutToolsCount: 12,
    toolsUnseenInLast7DaysCount: 7,
}

const randomImageUrl = "https://picsum.photos/500"

export const dummyEvents: Event[] = [
    {
        id: "1",
        timestamp: 1717000000,
        type: "tool_checkin",
        user: {
            id: "1",
            name: "John Doe",
            email: "john.doe@example.com",
            imageUrl: randomImageUrl,
        },
        tool: {
            id: "1",
            name: "Tool 1",
            description: "Tool 1 description",
            imageUrl: randomImageUrl,
            type: "Computer"
        },
        eventImageUrl: randomImageUrl,
    }
]